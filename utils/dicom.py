import pydicom
import numpy as np
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut


def load_image(image_path: str) -> Image.Image:
    """
    Load DICOM or PNG image with absolutely no data loss.

    - DICOM (.dcm): reads raw pixel data with modality LUT (slope/intercept),
      keeps 16-bit integer precision, returns PIL.Image in mode 'I;16'.
    - PNG (.png): reads back the exact 16-bit data (mode 'I;16') with no rescaling.
    """

    path = str(image_path)

    # ----- DICOM -----
    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path)
        arr = apply_modality_lut(ds.pixel_array, ds)  # Apply slope/intercept

        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = arr.max() - arr

        arr = np.asarray(arr, dtype=np.uint16)
        return Image.fromarray(arr, mode="I;16")

    # ----- PNG -----
    else:
        # Use a context manager and copy to detach from the file descriptor.
        with Image.open(path) as img:
            if img.mode != "I;16":
                raise ValueError(
                    f"Expected a 16-bit PNG, got {img.mode}. "
                    "Ensure conversion used mode='I;16'."
                )
            return img.copy()
