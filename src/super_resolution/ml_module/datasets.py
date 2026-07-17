from torch.utils.data import Dataset
import numpy as np
from math import cos, sin, pi
import torch

RACMO_TRAIN_MEAN = 14.94593
RACMO_TRAIN_STD = 29.616882
RACMO_TRAIN_VARIANCE = 877.1597

def standardize(data, mean=None, std=None, ownstd=True):
    # TO DO: check shapes when standardizing using precomputed mean and std (for GCM data)
    if ownstd:
        mean = np.nanmean(data, axis=(1,2), keepdims=True)
        std = np.nanstd(data, axis=(1,2), keepdims=True)
    ndata = (data - mean) / (std + 1e-8)
    # replace NaN from all-NaN channels (e.g. SST over land) with 0
    ndata = np.nan_to_num(ndata, nan=0.0)
    return ndata


class InferenceDataset(Dataset):
    def __init__(self, era5_patches, patches_metadata, ownstd=True):
        self.patches = era5_patches
        self.patches_metadata = patches_metadata
        self.ownstd = ownstd  # whether to compute own std or use precomputed mean/std
        self.target_train_mean = RACMO_TRAIN_MEAN
        self.target_train_std = RACMO_TRAIN_STD


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        input_patch = self.patches[idx]
        metadata = self.patches_metadata[idx]

        # drop channels with all NaN values (e.g. SST over land)
        input_patch = np.delete(input_patch, 3, axis=0)  # drop SST channel (index 3)

        # standardize input
        if self.ownstd:
            input_patch_stdz = standardize(input_patch, ownstd=self.ownstd)
        # else:
        #     # select mean and std for the current time (mean and std are precomputed for each variable and time)
        #     gcm_spatial_mean = self.gcm_space_mean.sel(valid_time=f"{year}-{month}-01").to_array().values
        #     gcm_spatial_std = self.gcm_space_std.sel(valid_time=f"{year}-{month}-01").to_array().values
        #     input_patch_stdz = standardize(input_patch, mean=gcm_spatial_mean, std=gcm_spatial_std, ownstd=self.ownstd)
        
        # get contextual statistics
        vect_std = np.nanstd(input_patch, axis=(1,2))
        vect_means = np.nanmean(input_patch, axis=(1,2))
        SpatialMean = vect_means.reshape(input_patch.shape[0], 1, 1)
        SpatialSTD = vect_std.reshape(input_patch.shape[0], 1, 1)

        # encode month
        month = str(metadata["time_idx"].astype('datetime64[D]')).split('-')[1]
        cosmonth = np.array([cos(2 * pi * (int(month) - 1) / 12)])
        sinmonth = np.array([sin(2 * pi * (int(month) - 1) / 12)])

        # broadcast to match the input patch shape
        cosvect = np.repeat(cosmonth, 1).reshape(1, 1, 1)
        sinvect = np.repeat(sinmonth, 1).reshape(1, 1, 1)

        # compose 1D input vector with spatial mean, spatial std, and month encoding
        input1d = np.concatenate((SpatialMean, SpatialSTD, cosvect, sinvect), axis=0)

        return input_patch_stdz, input1d
    

def signed_log1p(x, eps=1e-6):
    """Applies log1p to magnitude, preserves sign. Safe for negative values."""
    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)

def inverse_signed_log1p(x_transformed, eps=1e-6):
    """Inverse transform to recover original scale."""
    return torch.sign(x_transformed) * (torch.expm1(torch.abs(x_transformed)) - eps)

def unstandardize(data, mean=RACMO_TRAIN_MEAN, std=RACMO_TRAIN_STD):
    return data * std + mean

def standardize(data, mean=None, std=None, ownstd=True):
    # TO DO: check shapes when standardizing using precomputed mean and std (for GCM data)
    if ownstd:
        mean = np.nanmean(data, axis=(1,2), keepdims=True)
        std = np.nanstd(data, axis=(1,2), keepdims=True)
    ndata = (data - mean) / (std + 1e-8)
    # replace NaN from all-NaN channels (e.g. SST over land) with 0
    ndata = np.nan_to_num(ndata, nan=0.0)
    return ndata

# create mask
def createMask(true_smb):

    mask = np.empty((true_smb.shape[0], true_smb.shape[1]))
    for y in range(true_smb.shape[1]):
        for x in range(true_smb.shape[0]):
            pixelSMB = true_smb[x, y]
            if not np.any(pixelSMB):  # check if all zeros (then on sea)
                mask[x, y] = 0
            else:
                mask[x, y] = 1
    return mask


