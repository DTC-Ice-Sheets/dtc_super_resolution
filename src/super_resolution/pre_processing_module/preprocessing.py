""""This file contains the preprocessing pipeline for the RCM emulator."""
import xarray as xr
import numpy as np
from tqdm import tqdm

from super_resolution.pre_processing_module.reprojection import create_target_grid_from_RACMO, reproject_ERA5_to_RACMO_grid
from super_resolution.pre_processing_module.tiling import make_tiles, plotTiles
# from super_resolution.monitoring import setup_logging
import logging
# setup_logging()
logger = logging.getLogger(__name__)


def run_preprocessing(data_paths: dict) -> dict:
    """
    Run the preprocessing pipeline for the RCM emulator.

    Parameters:
    -----------
    data_paths: dict
        A dictionary containing paths to the downloaded ERA5 and RCM data.

    Returns:
    -------
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
    gcm_ds = load_era5_dataset(data_paths["era5"])
    rcm_ds = load_rcm_dataset(data_paths["rcm"])

    # Create target grid from RACMO
    # define the target coarser resolution, roughly matching ERA5 grid spacing at our target latitude of 75°S,
    resx_km = 14
    resy_km = 28
    gridx, gridy = create_target_grid_from_RACMO(rcm_ds, resx_km=resx_km, resy_km=resy_km)

    # Reproject ERA5 to RACMO grid
    era5_xy = reproject_ERA5_to_RACMO_grid(gcm_ds, gridx, gridy)

    # Prepare model inputs
    tiling_params = {
        "tile_size": 16,
        "context_pad_h": 16,
        "context_pad_w": 8,
        "stride_frac": 0.7,
        "log_": False}
    
    tiles = make_tiles(era5_xy, rcm_ds,
                        **tiling_params)  # Assuming tiling_params is defined elsewhere
    
    # fig = plotTiles(era5_xy, rcm_ds, tiles)
    # fig.savefig("tiles.png", dpi=300)

    input_patches, metadata = generate_era5_patches(era5_xy, tiles)

    return input_patches, metadata, tiles, rcm_ds, era5_xy


def generate_era5_patches(era5: xr.Dataset, tiles: list):
    """
    Generate ERA5 inputs for the model based on the tiles metadata.
    Skip tiles that contain only sea values.

    Parameters:
    -----------
    era5: xarray.Dataset 
        The ERA5 dataset.
    tiles: list
        A list of dictionaries containing metadata for each tile.

    Returns:
    --------
    (era5_patches, metadata): tuple 
        A tuple containing the ERA5 patches (numpy arrays) and their corresponding metadata (list of dicts).
    """
    era5_patches = []
    metadata = []
    
    era5_vars = [v for v in era5.data_vars
                if "x" in era5[v].dims and "y" in era5[v].dims]
    # time dimension
    era5_time = era5.valid_time.values
    # get numpy array (months, channels, height, width)
    era5_arr = np.stack([era5[v].compute().values for v in era5_vars], axis=1)
    era5_arr = np.nan_to_num(era5_arr, nan=0.0)  # replace NaN with 0

    # for each month in era5
    for t_idx in tqdm(range(len(era5_time)), desc="Processing ERA5 time steps"):
        era5_month_arr  = era5_arr[t_idx]   # (C, H_era5, W_era5)
        # iterate over the 506 tiles
        era5_patches_month, metadata_month = [], []
        for i, tile in enumerate(tiles):
            ey, ex = tile["era5_slice_y"],  tile["era5_slice_x"]

            # Extract the ERA5 patch corresponding to the tile
            era5_patch  = era5_month_arr[:, ey, ex]  # (C, tile_era5_h, tile_era5_w)
            
            # skip tiles that contain only sea values (all 0.0)
            sst = era5_patch[3, :, :]  # sea surface temperature 
            land_mask = (sst == 0.0)
            if land_mask.mean() < 0.5:  # if more than 95% of the tile is sea
                continue

            era5_patches_month.append(era5_patch)
            meta = {
                "time_idx": era5_time[t_idx],
                "tile_idx": i,
                "era5_slice_y": ey,
                "era5_slice_x": ex}
            metadata_month.append(meta)

        era5_patches.append(era5_patches_month)
        metadata.append(metadata_month)

    era5_patches = np.array(era5_patches)  # (T, N, C, H, W)    
    
    return era5_patches, metadata


def load_era5_dataset(era5_path: str):
    """
    Load ERA5 dataset from the given path.

    Args:
        era5_path (str): Path to the ERA5 dataset.  
    
    Returns:
        xarray.Dataset: Loaded ERA5 dataset.
    """
    return xr.open_dataset(era5_path)



def load_rcm_dataset(rcm_path: str):
    """
    Load RCM dataset from the given path.

    Args:
        rcm_path (str): Path to the RCM dataset.

    Returns:
        xarray.Dataset: Loaded RCM dataset.
    """
    racmo_ds = xr.open_zarr(rcm_path, consolidated=True)
    racmo_ds = racmo_ds.rio.write_crs("epsg:3031")
    return racmo_ds