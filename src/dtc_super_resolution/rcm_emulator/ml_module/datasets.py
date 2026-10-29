"""Dataset and helper functions for inference in the super-resolution model."""

from math import cos, pi, sin

import numpy as np
from torch.utils.data import Dataset

RACMO_TRAIN_MEAN = 14.94593
RACMO_TRAIN_STD = 29.616882
RACMO_TRAIN_VARIANCE = 877.1597


class InferenceDataset(Dataset):
    """Custom Dataset for inference, handling ERA5 patches and their metadata."""

    def __init__(self, era5_patches: np.ndarray, patches_metadata: list) -> None:
        super().__init__()
        self.patches = era5_patches
        self.patches_metadata = patches_metadata
        self.target_train_mean = RACMO_TRAIN_MEAN
        self.target_train_std = RACMO_TRAIN_STD

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the standardized input patch and its corresponding 1D input vector."""
        input_patch = self.patches[idx]
        metadata = self.patches_metadata[idx]

        # drop channels with all NaN values (e.g. SST over land)
        input_patch = np.nan_to_num(input_patch, nan=0.0)  # replace NaN with 0

        # standardize the input
        input2d_patch_stdz = standardize(input_patch)

        # get contextual statistics
        vect_std = np.nanstd(input_patch, axis=(1, 2))
        vect_means = np.nanmean(input_patch, axis=(1, 2))
        spatial_mean = vect_means.reshape(input_patch.shape[0], 1, 1)
        spatial_std = vect_std.reshape(input_patch.shape[0], 1, 1)

        # encode month
        month = str(metadata["time_idx"].astype("datetime64[D]")).split("-")[1]
        cosmonth = np.array([cos(2 * pi * (int(month) - 1) / 12)])
        sinmonth = np.array([sin(2 * pi * (int(month) - 1) / 12)])

        # broadcast to match the input patch shape
        cosvect = np.repeat(cosmonth, 1).reshape(1, 1, 1)
        sinvect = np.repeat(sinmonth, 1).reshape(1, 1, 1)

        # compose 1D input vector with spatial mean, spatial std, and month encoding
        input1d = np.concatenate((spatial_mean, spatial_std, cosvect, sinvect), axis=0)

        return input2d_patch_stdz.astype(np.float32), input1d.astype(np.float32)


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize the data using either its own mean and std or provided mean and std."""
    mean = np.nanmean(data, axis=(1, 2), keepdims=True)
    std = np.nanstd(data, axis=(1, 2), keepdims=True)
    ndata = (data - mean) / (std + 1e-8)

    return np.nan_to_num(ndata, nan=0.0)


def unstandardize(data: np.ndarray, mean: float = RACMO_TRAIN_MEAN, std: float = RACMO_TRAIN_STD) -> np.ndarray:
    """Unstandardize the data using provided mean and std."""
    return data * std + mean


def create_mask(true_smb: np.ndarray) -> np.ndarray:
    """Create a mask for the true SMB data, marking sea pixels as 0 and land pixels as 1."""
    mask = np.empty((true_smb.shape[0], true_smb.shape[1]))
    for y in range(true_smb.shape[1]):
        for x in range(true_smb.shape[0]):
            pixel_smb = true_smb[x, y]
            if not np.any(pixel_smb):  # check if all zeros (then on sea)
                mask[x, y] = 0
            else:
                mask[x, y] = 1
    return mask
