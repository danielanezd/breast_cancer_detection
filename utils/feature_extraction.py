import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from PIL import Image
import os

from utils.constants import MODELS_OUTPUT_PATH, TEXT_CLEANED_IMAGES_ABS_PATH

FEATURE_COLUMNS = ["mean_intensity", "std_intensity", "width", "height"]


def build_route(df, images_base_path=None):
    """
    Build the full image path for each row in the DataFrame and filter out non-existent files.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'image_path' column containing relative paths.
        
    Returns:
        pd.DataFrame: Filtered dataframe containing only rows with existing image files.
    """
    images_path = TEXT_CLEANED_IMAGES_ABS_PATH if images_base_path is None else images_base_path

    df['full_image_path'] = df['image_path'].apply(
        lambda x: os.path.normpath(os.path.join(images_path, x))
    )
    df['exists'] = df['full_image_path'].apply(os.path.exists)
    return df[df['exists']].copy()

def extract_features(df: pd.DataFrame, save_name: str = None) -> pd.DataFrame:
    """
    Extract basic image features from the input DataFrame and optionally save the results.

    Args:
        df (pd.DataFrame): Input dataframe with image paths.
        save_name (str, optional): If provided, saves the result as a .parquet file with this base name.

    Returns:
        pd.DataFrame: DataFrame with extracted image features merged.
    """
    features_list = []

    for path in tqdm(df["full_image_path"], desc="Extracting features"):
        try:
            if str(path).lower().endswith(".dcm"):
                # Read DICOM
                dicom = pydicom.dcmread(path)
                img_array = dicom.pixel_array.astype(np.float32)
            else:
                # Read PNG/JPEG
                img = Image.open(path).convert("L")
                img_array = np.asarray(img, dtype=np.float32)

            if img_array.size == 0:
                raise ValueError("Empty image.")

            # Normalize to [0, 1] using global scaling (safe for both DICOM and PNG)
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)

            mean_intensity = img_array.mean()
            std_intensity = img_array.std()
            height, width = img_array.shape

            features_list.append({
                "full_image_path": path,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "width": width,
                "height": height
            })
        except Exception as e:
            print(f"[WARN] Error with image {path}: {e}")

    features_df = pd.DataFrame(features_list)
    merged_df = df.merge(features_df, on="full_image_path", how="inner")

    #if save_name:
    #    output_path = os.path.join(MODELS_OUTPUT_PATH, f"{save_name}_features.parquet")
    #    merged_df.to_parquet(output_path, index=False)
    #    print(f"Features saved to: {output_path}")

    return merged_df
