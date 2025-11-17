import os
from pathlib import Path
import shutil
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures

from constants import CLEANED_IMAGES_ABS_PATH, IMAGES_ABS_PATH
from dicom import load_image


def convert_and_save_image(args):
    img_path, source_dir, output_dir = args
    try:
        rel_path = img_path.relative_to(source_dir)
        out_path = (output_dir / rel_path).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Use LUT-aware loader and preserve 16-bit depth in exported PNGs
        image = load_image(str(img_path))
        image.save(out_path, format="PNG")

        return (str(img_path), "ok", "")
    except Exception as e:
        return (str(img_path), "error", str(e))


def preconvert_images_parallel(source_dir: str, output_dir: str, max_workers: int = 2):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    supported_exts = {".dcm", ".jpg", ".jpeg", ".png"}

    image_paths = [fp for fp in source_dir.rglob("*") if fp.suffix.lower() in supported_exts]
    print(f"Found {len(image_paths)} image(s) to process.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = ((img_path, source_dir, output_dir) for img_path in image_paths)
        results = list(tqdm(executor.map(convert_and_save_image, args), total=len(image_paths), desc="Converting"))

    # Optional logging
    import pandas as pd
    df = pd.DataFrame(results, columns=["file", "status", "error"])
    log_path = output_dir / "conversion_log.csv"
    df.to_csv(log_path, index=False)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Parallel preconversion of DICOM and other images to PNG.")
    parser.add_argument("--source", "-s", type=str, required=False, help="Source directory with raw images.")
    parser.add_argument("--output", "-o", type=str, required=False, help="Destination directory for converted PNGs.")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: 75%% of cores).")
    args = parser.parse_args()

    source_root = args.source if args.source else IMAGES_ABS_PATH
    output_root = args.output if args.output else CLEANED_IMAGES_ABS_PATH
    num_workers = args.workers or max(1, int(multiprocessing.cpu_count() * 0.75))

    preconvert_images_parallel(source_root, output_root, max_workers=num_workers)
