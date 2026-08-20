"""Preprocessing pipeline for the RCM emulator."""

# from super_resolution.monitoring import setup_logging
import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

from super_resolution.pre_processing_module.reprojection import (
    create_target_grid_from_racmo,
    reproject_era5_to_racmo_grid,
)
from super_resolution.pre_processing_module.tiling import make_tiles

# setup_logging()
logger = logging.getLogger(__name__)

EXPECTED_ERA5_VARIABLES = ["d2m", "msl", "sp", "sst", "t2m", "tp", "u10", "v10"]


def run_preprocessing(gcm_zarr_path: str, rcm_zarr_path: str) -> tuple[np.ndarray, list, list, xr.Dataset, xr.Dataset]:
    """
    Run the preprocessing pipeline for the RCM emulator.

    Parameters
    ----------
    gcm_zarr_path: str
        Path to the GCM data.
    rcm_zarr_path: str
        Path to the RCM data.

    Returns
    -------
    tuple[np.ndarray, list, list, xr.Dataset, xr.Dataset]
        A tuple containing:
            input_patches: np.ndarray
                A numpy array containing the preprocessed ERA5 patches ready for model input.
            metadata: list
                A list of dictionaries containing metadata for each ERA5 patch.
            tiles: list
                A list of dictionaries containing metadata for each tile.
            rcm_ds: xarray.Dataset
                The RCM dataset.
            era5_xy: xarray.Dataset
                The ERA5 dataset reprojected to the RCM grid.
    """
    # Load ERA5 and RCM datasets
    gcm_ds = load_era5_dataset(gcm_zarr_path)
    rcm_ds = load_rcm_dataset(rcm_zarr_path)

    # validate ERA5 dataset
    validate_era5_dataset(gcm_ds)

    # Create target grid from RACMO
    # define the target coarser resolution, roughly matching ERA5 grid spacing at our target latitude of 75°S,
    resx_km = 14
    resy_km = 28
    gridx, gridy = create_target_grid_from_racmo(rcm_ds, resx_km=resx_km, resy_km=resy_km)

    # Reproject ERA5 to RACMO grid
    era5_xy = reproject_era5_to_racmo_grid(gcm_ds, gridx, gridy)

    # Prepare model inputs
    tiling_params = {"tile_size": 16, "context_pad_h": 16, "context_pad_w": 8, "stride_frac": 0.7, "log_": False}

    tiles = make_tiles(era5_xy, rcm_ds, **tiling_params)

    input_patches, metadata = generate_era5_patches(era5_xy, tiles)

    return input_patches, metadata, tiles, rcm_ds, era5_xy


def validate_era5_dataset(era5: xr.Dataset) -> None:
    """
    Validate the ERA5 dataset to ensure it contains each of the required layers.

    Parameters
    ----------
    era5: xr.Dataset
        The ERA5 dataset to validate.

    Raises
    ------
    ValueError
        If any of the required variables are missing from the dataset.
    """
    for v in EXPECTED_ERA5_VARIABLES:
        if v not in era5.data_vars:
            raise ValueError(
                f"Variable '{v}' not found in ERA5 dataset. "
                f"Expected variables: {EXPECTED_ERA5_VARIABLES}. "
                f"Present variables: {list(era5.data_vars)}."
            )

    # Check that the dataset supports all required dimensions with non-zero sizes
    len_era5_x = len(era5.x)
    len_era5_y = len(era5.y)
    len_era5_time = len(era5.time)
    if any([len_era5_x <= 0, len_era5_y <= 0, len_era5_time <= 0]):
        raise ValueError(
            f"ERA5 dataset must have x, y & time dimensions greater than 0, "
            f"found: x={len_era5_x}, y={len_era5_y}, time={len_era5_time}."
            " please increase the area of your search polygon or timespan accordingly."
        )


def generate_era5_patches(era5: xr.Dataset, tiles: list) -> tuple[np.ndarray, list]:
    """
    Generate ERA5 inputs for the model based on the tiles metadata.

    Skip tiles that contain only sea values.

    Parameters
    ----------
    era5: xr.Dataset
        The ERA5 dataset.
    tiles: list
        A list of dictionaries containing metadata for each tile.

    Returns
    -------
    tuple[np.ndarray, list]
        A tuple containing the ERA5 patches (numpy arrays) and their corresponding metadata (list of dicts).
    """
    era5_patches = []
    metadata = []

    # time dimension
    era5_time = era5.time.values

    # get numpy array (months, channels, height, width)
    era5_arr = np.stack([era5[v].compute().values for v in EXPECTED_ERA5_VARIABLES], axis=1)
    era5_arr = np.nan_to_num(era5_arr, nan=0.0)  # replace NaN with 0

    # for each month in era5
    for t_idx in tqdm(range(len(era5_time)), desc="Processing ERA5 time steps"):
        era5_month_arr = era5_arr[t_idx]  # (C, H_era5, W_era5)
        # iterate over the 506 tiles
        era5_patches_month, metadata_month = [], []
        for i, tile in enumerate(tiles):
            ey, ex = tile["era5_slice_y"], tile["era5_slice_x"]

            # Extract the ERA5 patch corresponding to the tile
            era5_patch = era5_month_arr[:, ey, ex]  # (C, tile_era5_h, tile_era5_w)

            # skip tiles that contain only sea values (all 0.0)
            sst = era5_patch[3, :, :]  # sea surface temperature
            land_mask = sst == 0.0
            if land_mask.mean() < 0.1:  # if more than 99% of the tile is sea
                continue

            era5_patches_month.append(era5_patch)
            meta = {"time_idx": era5_time[t_idx], "tile_idx": i, "era5_slice_y": ey, "era5_slice_x": ex}
            metadata_month.append(meta)

        era5_patches.append(era5_patches_month)
        metadata.append(metadata_month)

    era5_patches = np.array(era5_patches)  # (T, N, C, H, W)

    return era5_patches, metadata


def load_era5_dataset(era5_path: str) -> xr.Dataset:
    """
    Load ERA5 dataset from the given path.

    Parameters
    ----------
    era5_path : str
        Path to the ERA5 dataset.

    Returns
    -------
    xr.Dataset
        Loaded ERA5 dataset.
    """
    return xr.open_dataset(era5_path)


def load_rcm_dataset(rcm_path: str) -> xr.Dataset:
    """
    Load RCM dataset from the given path.

    Parameters
    ----------
    rcm_path : str
        Path to the RCM dataset.

    Returns
    -------
    xr.Dataset
        Loaded RCM dataset.
    """
    racmo_ds = xr.open_zarr(rcm_path, consolidated=True)
    return racmo_ds.rio.write_crs("epsg:3031")
