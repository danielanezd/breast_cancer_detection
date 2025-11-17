import torch
import numpy as np


class ToFloatTensor16Bit:
    def __call__(self, pic):
        arr = np.array(pic, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 65535.0  # normalize 16-bit range

        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        tensor = tensor.repeat(3, 1, 1)  # [3,H,W] for pretrained CNNs
        return tensor
