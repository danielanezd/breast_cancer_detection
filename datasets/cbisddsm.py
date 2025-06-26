from torch.utils.data import Dataset
import torch
from pathlib import Path
import pyarrow.parquet as pq
from utils.dicom import load_image
from utils.constants import TEXT_CLEANED_IMAGES_ABS_PATH, LABEL_MAP
import os

class CBISDDSMDataset(Dataset):
    def __init__(self, parquet_path, transform=None, label_map=None, images_base_path=None, multi_view=True):
        self.table = pq.read_table(parquet_path).to_pandas()
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
        self.images_base_path = Path(images_base_path or TEXT_CLEANED_IMAGES_ABS_PATH)
        self.multi_view = multi_view

        if multi_view:
            self.samples = []
            for _, row in self.table.iterrows():
                for view in ["image", "cropped", "roi"]:
                    self.samples.append((row, view))
        else:
            self.samples = [(row, "image") for _, row in self.table.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row, view = self.samples[idx]
        path = row[f"{view}_path"].lstrip('/')
        img = load_image(self.images_base_path / path)
        if self.transform:
            img = self.transform(img)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        label = self.label_map[row["pathology"]]
        return img, torch.tensor(label, dtype=torch.long)
