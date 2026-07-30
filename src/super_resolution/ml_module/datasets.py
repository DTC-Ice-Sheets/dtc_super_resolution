"""Dataset and helper functions for inference in the super-resolution model."""

from math import cos, pi, sin

import numpy as np
import torch
from torch.utils.data import Dataset

RACMO_TRAIN_MEAN = 14.94593
RACMO_TRAIN_STD = 29.616882
RACMO_TRAIN_VARIANCE = 877.1597


class InferenceDataset(Dataset):
    """Custom Dataset for inference, handling ERA5 patches and their metadata."""

    def __init__(self, era5_patches: np.ndarray, patches_metadata: list, ownstd: bool = True) -> None:
        super().__init__()
        self.patches = era5_patches
        self.patches_metadata = patches_metadata
        self.ownstd = ownstd  # whether to compute own std or use precomputed mean/std
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
        input_patch = np.delete(input_patch, 3, axis=0)  # drop SST channel (index 3)

        # standardize input
        if self.ownstd:
            input_patch_stdz = standardize(input_patch, ownstd=self.ownstd)
        # else:
        #     # select mean and std for the current time (mean and std are precomputed for each variable and time)
        #     gcm_spatial_mean = self.gcm_space_mean.sel(time=f"{year}-{month}-01").to_array().values
        #     gcm_spatial_std = self.gcm_space_std.sel(time=f"{year}-{month}-01").to_array().values
        #     input_patch_stdz = standardize(input_patch,
        #                                       mean=gcm_spatial_mean,
        #                                       std=gcm_spatial_std, ownstd=self.ownstd)

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

        return input_patch_stdz, input1d


def signed_log1p(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply log1p to magnitude, preserves sign. Safe for negative values."""
    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)


def inverse_signed_log1p(x_transformed: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse transform to recover original scale."""
    return torch.sign(x_transformed) * (torch.expm1(torch.abs(x_transformed)) - eps)


def standardize(
    data: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None, ownstd: bool = True
) -> np.ndarray:
    """Standardize the data using either its own mean and std or provided mean and std."""
    # TO DO: check shapes when standardizing using precomputed mean and std (for GCM data)
    if ownstd:
        mean = np.nanmean(data, axis=(1, 2), keepdims=True)
        std = np.nanstd(data, axis=(1, 2), keepdims=True)
    ndata = (data - mean) / (std + 1e-8)
    # replace NaN from all-NaN channels (e.g. SST over land) with 0
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
