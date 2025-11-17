from pathlib import Path
from PIL import Image
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

from constants import CLEANED_IMAGES_ABS_PATH, TEXT_CLEANED_IMAGES_ABS_PATH
from dicom import load_image  # Import your 16-bit preserving load_image


def normalize_to_8bit_for_ocr(image_16bit: Image.Image) -> Image.Image:
    """
    Convert 16-bit grayscale image to 8-bit RGB for OCR processing ONLY.
    This is a temporary conversion - we won't save this, only use for detection.
    
    Args:
        image_16bit: PIL Image in mode 'I;16'
    
    Returns:
        PIL Image in mode 'RGB' suitable for EasyOCR
    """
    # Convert to numpy array (keeps uint16)
    arr = np.array(image_16bit)
    
    # Normalize to 0-255 range for visualization
    # Use percentile-based normalization for better contrast
    p2, p98 = np.percentile(arr, (2, 98))
    arr_clipped = np.clip(arr, p2, p98)
    
    # Scale to 8-bit range
    if p98 > p2:
        arr_normalized = ((arr_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        arr_normalized = np.zeros_like(arr, dtype=np.uint8)
    
    # Convert to RGB for EasyOCR (it expects 3 channels)
    arr_rgb = np.stack([arr_normalized] * 3, axis=-1)
    
    return Image.fromarray(arr_rgb, mode='RGB')


def detect_best_ocr_result(image_16bit: Image.Image, reader, conf_threshold=0.5):
    """
    Detect text in different image orientations.
    Works with 16-bit image by temporarily converting to 8-bit for OCR only.
    
    Returns:
        tuple: (best_result, best_transform, original_size)
    """
    # Store original size to map coordinates back
    original_size = image_16bit.size
    
    # Create 8-bit version for OCR
    image_8bit = normalize_to_8bit_for_ocr(image_16bit)
    
    variants = {
        "original": (image_8bit, image_16bit),
        "flipped": (
            image_8bit.transpose(Image.FLIP_LEFT_RIGHT),
            image_16bit.transpose(Image.FLIP_LEFT_RIGHT)
        ),
        "rot45": (
            image_8bit.rotate(45, expand=True),
            image_16bit.rotate(45, expand=True)
        ),
        "rot135": (
            image_8bit.rotate(135, expand=True),
            image_16bit.rotate(135, expand=True)
        ),
    }

    best_result = []
    best_score = 0
    best_transform = "original"
    best_16bit_variant = image_16bit

    for name, (variant_8bit, variant_16bit) in variants.items():
        # OCR on 8-bit version only
        ocr_result = reader.readtext(np.array(variant_8bit))
        score = sum(conf for _, _, conf in ocr_result if conf >= conf_threshold)
        if score > best_score:
            best_result = ocr_result
            best_score = score
            best_transform = name
            best_16bit_variant = variant_16bit

    return best_result, best_transform, best_16bit_variant


def crop_fixed_margins(image: Image.Image, crop_px: int = 60) -> Image.Image:
    """
    Crop fixed margins from image. Works with any bit depth.
    """
    w, h = image.size
    left = crop_px
    right = w - crop_px
    top = crop_px
    bottom = h - crop_px

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


def mask_text_16bit(image_16bit: Image.Image, reader, fill_value=0, draw_outline=False):
    """
    Detect and mask text regions in a 16-bit grayscale image.
    
    Args:
        image_16bit: PIL Image in mode 'I;16'
        reader: EasyOCR reader instance
        fill_value: Value to fill masked regions (default 0 = black)
        draw_outline: If True, draw outline around masked regions (for debugging)
    
    Returns:
        tuple: (masked_image_16bit, text_found, transform_used)
    """
    # Detect text using 8-bit version, but get back transformed 16-bit image
    results, transform, transformed_16bit = detect_best_ocr_result(image_16bit, reader)
    
    # Convert 16-bit to numpy for manipulation
    arr_16bit = np.array(transformed_16bit, dtype=np.uint16)
    
    text_found = False

    for (bbox, text, conf) in results:
        if conf < 0.5:
            continue
        text_found = True
        
        # Get bounding box coordinates
        pts = [tuple(map(int, point)) for point in bbox]
        x_coords, y_coords = zip(*pts)
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        
        # Add padding
        pad_x = int(bbox_width * 0.25)
        pad_y = int(bbox_height * 0.25)

        x_min = max(min(x_coords) - pad_x, 0)
        x_max = min(max(x_coords) + pad_x, arr_16bit.shape[1])
        y_min = max(min(y_coords) - pad_y, 0)
        y_max = min(max(y_coords) + pad_y, arr_16bit.shape[0])

        # Mask the region in the 16-bit array
        arr_16bit[y_min:y_max, x_min:x_max] = fill_value

    # Convert back to PIL Image in 16-bit mode
    masked_image = Image.fromarray(arr_16bit, mode='I;16')
    
    # Undo transformation to get back to original orientation
    if transform == "flipped":
        masked_image = masked_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == "rot45":
        masked_image = masked_image.rotate(-45, expand=True)
        # Crop back to original size after rotation
        w_orig, h_orig = image_16bit.size
        w_rot, h_rot = masked_image.size
        left = (w_rot - w_orig) // 2
        top = (h_rot - h_orig) // 2
        masked_image = masked_image.crop((left, top, left + w_orig, top + h_orig))
    elif transform == "rot135":
        masked_image = masked_image.rotate(-135, expand=True)
        # Crop back to original size after rotation
        w_orig, h_orig = image_16bit.size
        w_rot, h_rot = masked_image.size
        left = (w_rot - w_orig) // 2
        top = (h_rot - h_orig) // 2
        masked_image = masked_image.crop((left, top, left + w_orig, top + h_orig))

    return masked_image, text_found, transform


def save_16bit_png(image: Image.Image, output_path: Path):
    """
    Save image as 16-bit PNG, preserving all data.
    
    Args:
        image: PIL Image in mode 'I;16'
        output_path: Path to save the image
    """
    if image.mode != 'I;16':
        raise ValueError(f"Expected mode 'I;16', got {image.mode}")
    
    # PIL's save with format="PNG" will preserve I;16 mode
    image.save(output_path, format="PNG")


def clean_images_in_directory(source_root: str, output_root: str, use_gpu: bool = False, crop_px: int = 60):
    """
    Process all medical images in a directory, masking text while preserving 16-bit quality.
    
    Args:
        source_root: Directory containing 16-bit PNG or DICOM images
        output_root: Directory to save cleaned images
        use_gpu: Whether to use GPU for OCR
        crop_px: Pixels to crop from borders after masking text
    """
    supported_exts = {".png", ".dcm"}  # Support both PNG and DICOM
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_folder = source_root.name
    log_filename = f"log_{timestamp}_{source_folder}.csv"
    log_path = output_root / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize log
    pd.DataFrame(columns=["file_path", "status", "error", "bit_depth", "transform"]).to_csv(log_path, index=False)

    files_to_process = [fp for fp in source_root.rglob("*") if fp.suffix.lower() in supported_exts]
    print(f"Processing {len(files_to_process)} images...")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"GPU enabled: {use_gpu}")

    reader = easyocr.Reader(['en'], gpu=use_gpu)
    results = []

    for fp in tqdm(files_to_process, desc="Processing", unit="img"):
        try:
            # Load image using your quality-preserving function
            image_16bit = load_image(str(fp))
            
            # Verify it's 16-bit
            if image_16bit.mode != 'I;16':
                raise ValueError(f"Image is not in 16-bit mode (I;16), got {image_16bit.mode}")
            
            # Detect and mask text
            masked_img, found_text, transform = mask_text_16bit(image_16bit, reader)
            
            # Verify output is still 16-bit
            if masked_img.mode != 'I;16':
                raise ValueError(f"Masked image lost 16-bit mode, got {masked_img.mode}")

            # Determine output path (always save as .png)
            rel_path = fp.relative_to(source_root)
            output_path = (output_root / rel_path).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if found_text:
                # Crop margins and save
                masked_img = crop_fixed_margins(masked_img, crop_px=crop_px)
                save_16bit_png(masked_img, output_path)
                status = "processed"
            else:
                # No text found, save original (still in 16-bit)
                save_16bit_png(image_16bit, output_path)
                status = "copied"

            results.append({
                "file_path": str(fp),
                "status": status,
                "error": "",
                "bit_depth": "16-bit",
                "transform": transform if found_text else "none"
            })
            
        except Exception as e:
            results.append({
                "file_path": str(fp),
                "status": "failed",
                "error": str(e),
                "bit_depth": "unknown",
                "transform": "none"
            })
            tqdm.write(f"Error processing {fp}: {e}")

    # Save log
    pd.DataFrame(results).to_csv(log_path, mode='a', header=False, index=False)
    
    # Print summary
    df_results = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Successfully processed: {len(df_results[df_results['status'] == 'processed'])}")
    print(f"Copied (no text): {len(df_results[df_results['status'] == 'copied'])}")
    print(f"Failed: {len(df_results[df_results['status'] == 'failed'])}")
    print(f"\nLog saved to: {log_path}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and mask text in medical images while preserving 16-bit quality."
    )
    parser.add_argument(
        "--source", "-s", 
        required=False, 
        type=str,
        help="Path to the directory containing 16-bit PNG or DICOM images."
    )
    parser.add_argument(
        "--output", "-o", 
        required=False, 
        type=str,
        help="Path to the output directory for cleaned images."
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU usage for OCR (recommended for large datasets)."
    )
    parser.add_argument(
        "--crop", 
        type=int, 
        default=60, 
        help="Pixels to crop from borders after masking text (default: 60)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_root = args.source if args.source else CLEANED_IMAGES_ABS_PATH
    output_root = args.output if args.output else TEXT_CLEANED_IMAGES_ABS_PATH

    print(f"\n{'='*60}")
    print("16-BIT MEDICAL IMAGE TEXT CLEANING")
    print(f"{'='*60}\n")

    clean_images_in_directory(
        source_root=source_root,
        output_root=output_root,
        use_gpu=args.gpu,
        crop_px=args.crop
    )