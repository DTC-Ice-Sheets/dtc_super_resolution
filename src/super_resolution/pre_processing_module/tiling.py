import json
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from tqdm import tqdm 
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr

import logging
# from super_resolution.monitoring import setup_logging
# setup_logging()
logger = logging.getLogger(__name__)

def make_tiles(era5: xr.DataArray, 
               racmo: xr.DataArray, 
               tile_size: int, 
               context_pad_h: int, 
               context_pad_w: int, 
               stride_frac: float = 1.0,
               log_: bool = False) -> list:
    """"
    Generate tiles for training or inference.
    Each tile consists of a RACMO patch and a corresponding ERA5 patch.
    
    Parameters
    ----------
    racmo : xarray.DataArray
        RACMO data array (2D) in EPSG:3031 projection.
    era5 : xarray.DataArray
        ERA5 data array (2D) in EPSG:3031 projection.
    tile_size : int
        Size of the tile in pixels.
    context_pad_h : int
        Height of the era5 context padding in pixels.
    context_pad_w : int
        Width of the era5 context padding in pixels.
    stride_frac : float
        Fraction of tile size used as stride.
        1.0 → non-overlapping,  0.5 → 50 % overlap, etc.

    Returns
    -------
    tiles : list of dicts
    """
    # ── Resolutions (metres) ──────────────────────────────────────────────────────
    racmo_res = abs(racmo.rio.resolution()[0])          # 2000 m
    era5_res_x = abs(era5.rio.resolution()[0])          # 14000 m
    era5_res_y = abs(era5.rio.resolution()[1])          # 28000 m

    scale_x = int(round(era5_res_x / racmo_res))           # 7
    scale_y = int(round(era5_res_y / racmo_res))           # 14

    # compute the tile sizes from the input params
    tile_racmo_h = tile_size * scale_y                # RACMO rows  (448 for E=16)
    tile_racmo_w = tile_size * scale_x * 2            # RACMO cols  (448 for E=16)
    tile_era5_h  = tile_size                          # ERA5 rows
    tile_era5_w  = tile_size * 2                      # ERA5 cols

    # NOT NEEDED IN OUR CASE: ERA5 is already clipped to RACMO extent
    # racmo_ds, era5_xy = clip_data_to_intersection(racmo_ds, era5_xy, scale_x, scale_y)

    tiles = generate_tile_indices(racmo, era5,
                                tile_racmo_h, tile_racmo_w,
                                scale_x, scale_y,
                                stride_frac=stride_frac,
                                context_pad=(context_pad_h, context_pad_w))

    if log_:
        era5_patch_h = tile_era5_h + 2 * context_pad_h
        era5_patch_w = tile_era5_w + 2 * context_pad_w

        logger.info("\n> Context padding:")
        logger.info(f"\tHEIGHT: {context_pad_h} = {context_pad_h * era5_res_y / 1e3:.0f} km top/bottom")
        logger.info(f"\tWIDTH: {context_pad_w} = {context_pad_w * era5_res_x / 1e3:.0f} km left/right")
        logger.info(f"> Total tiles: {len(tiles)}")
        logger.info(f"> Patch sizes:")
        logger.info(f"\tERA5  : {era5_patch_h} x {era5_patch_w} px  "
            f"({era5_patch_h * era5_res_y / 1e3:.0f} x {era5_patch_w * era5_res_x / 1e3:.0f} km)")
        logger.info(f"\tRACMO : {tile_racmo_h} x {tile_racmo_w} px  "
            f"({tile_racmo_h * racmo_res / 1e3:.0f} x {tile_racmo_w * racmo_res / 1e3:.0f} km)")
    

    return tiles



def generate_tile_indices(racmo: xr.DataArray, era5: xr.DataArray, tile_racmo_h: int, tile_racmo_w: int,
                          scale_x: int, scale_y: int, stride_frac: float = 1.0, context_pad: tuple[int, int] = (0, 0)) -> list:
    """
    Generate paired RACMO / ERA5 index slices for all tiles.

    Because both grids are co-registered (same origin, ERA5 = scale x RACMO),
    ERA5 pixel k maps to RACMO pixels [k*scale, (k+1)*scale).
    The stride is snapped to a multiple of the scale factors so tiles
    always align exactly.

    Parameters
    ----------
    stride_frac : float
        Fraction of tile size used as stride.
        1.0 → non-overlapping,  0.5 → 50 % overlap, etc.
    context_pad : tuple[int, int]
        Extra ERA5 pixels added on each side of the ERA5 patch (height, width),
        providing spatial context beyond the RACMO target region.
        E.g. context_pad=(2, 2) gives an ERA5 patch of
        (tile_era5_h + 4) x (tile_era5_w + 4).
        Tiles where the padded ERA5 patch would exceed the array bounds
        are dropped.

    Returns
    -------
    list of dicts with:
        racmo_slice_y/x  - isel slices into racmo_clip
        era5_slice_y/x   - isel slices into era5_clip (padded)
    """
    tile_era5_h = tile_racmo_h // scale_y   # exact because tile_racmo = E * scale
    tile_era5_w = tile_racmo_w // scale_x

    # Snap stride to a multiple of scale factors
    stride_h = max(scale_y, int(tile_racmo_h * stride_frac // scale_y) * scale_y)
    stride_w = max(scale_x, int(tile_racmo_w * stride_frac // scale_x) * scale_x)

    H = racmo.rio.shape[0]
    W = racmo.rio.shape[1]
    H_era5 = era5.rio.shape[0]
    W_era5 = era5.rio.shape[1]

    tiles = []
    for i in range(0, H - tile_racmo_h + 1, stride_h):
        for j in range(0, W - tile_racmo_w + 1, stride_w):
            ei = i // scale_y
            ej = j // scale_x

            # ERA5 slice with context padding
            ei_start = ei - context_pad[0]
            ei_end = ei + tile_era5_h + context_pad[0]
            ej_start = ej - context_pad[1]
            ej_end = ej + tile_era5_w + context_pad[1]

            # Drop tiles where the padded ERA5 patch goes out of bounds
            if ei_start < 0 or ej_start < 0 or ei_end > H_era5 or ej_end > W_era5:
                continue

            tiles.append({
                "racmo_slice_y": slice(i, i + tile_racmo_h),
                "racmo_slice_x": slice(j, j + tile_racmo_w),
                "era5_slice_y":  slice(ei_start, ei_end),
                "era5_slice_x":  slice(ej_start, ej_end),
            })

    return tiles


def save_tiles_to_disk(era5: xr.DataArray, 
                       racmo_files: list, 
                       tiles: list, 
                       output_dir: str, 
                       max_nan_frac: float = 0.95) -> None:
    """
    For each RACMO file, we iterate over the tiles and save the corresponding 
    RACMO and ERA5 patches to disk.
    Tiles are stored as .npz files:  tile_<row>_<col>_<year>.npz with keys era5 and racmo.
    We filter out ocean-only tiles before saving using the RACMO land mask (NaN fraction check).
    
    Parameters
    ----------
    tiles : list of dicts
        Each dict contains slices for RACMO and ERA5 tiles.
    output_dir : str
        Path to the output directory.
    max_nan_frac : float, optional
        Maximum fraction of NaN values allowed in a tile.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ── ERA5 variables to include  ──────────────────────
    era5_vars = [v for v in era5.data_vars
                if "x" in era5[v].dims and "y" in era5[v].dims]

    # ── RACMO target variable ─────────────────────────────────────────────────────
    racmo_var = "surface_mass_balance_glaciated"

    saved = 0
    skipped_nan = 0

    for racmo_file in sorted(racmo_files):
        year_ds = xr.open_zarr(racmo_file).rio.write_crs("epsg:3031")

        # clip to the intersection area
        # year_ds_clip = year_ds.rio.clip_box(xmin, ymin, xmax, ymax)

        # ── Load the full RACMO year into memory once  ──
        racmo_year_arr = year_ds[racmo_var].compute().values
        # shape: (n_months, H_racmo, W_racmo)

        # ── Load all ERA5 time steps that overlap this year ───────────────────────
        racmo_times = year_ds.time.values
        era5_year = era5.sel(
            valid_time=racmo_times, method="nearest"
        )
        era5_year_arr = np.stack([era5_year[v].compute().values for v in era5_vars], axis=1)
        # shape: (n_months, C, H_era5, W_era5)

        for t_idx in tqdm(range(len(racmo_times)), desc="Tiling months"):
            racmo_month_arr = racmo_year_arr[t_idx]  # (H_racmo, W_racmo)
            era5_month_arr  = era5_year_arr[t_idx]   # (C, H_era5, W_era5)

            for i,t in enumerate(tiles):
                ry, rx = t["racmo_slice_y"], t["racmo_slice_x"]
                ey, ex = t["era5_slice_y"],  t["era5_slice_x"]

                racmo_patch = racmo_month_arr[ry, rx]    # (tile_racmo_h, tile_racmo_w)
                era5_patch  = era5_month_arr[:, ey, ex]  # (C, tile_era5_h, tile_era5_w)

                # Skip tiles with too many NaNs (ocean / no data)
                nan_frac = np.isnan(racmo_patch).mean()
                zero_frac = (racmo_patch == 0.0).mean()
                null_data = nan_frac + zero_frac

                if null_data > max_nan_frac:
                    skipped_nan += 1
                    continue

                row = ry.start // 224       # 224 = tile_size * scale_factor_7 * 2 
                col = rx.start // 224       # 224 = tile_size * scale_factor_14  
                ts  = pd.Timestamp(racmo_times[t_idx])
                fname = os.path.join(output_dir,
                                    f"tile_r{row:02d}_c{col:02d}_{ts.year}_{ts.month:02d}.npz")
                np.savez_compressed(fname, era5=era5_patch, racmo=racmo_patch, tile_id=i)
                saved += 1

        year_ds.close()

    return



def clip_data_to_intersection(racmo: xr.DataArray, era5: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Clip RACMO and ERA5 datasets to their intersection.

    Because both grids are co-registered (same origin, ERA5 = scale x RACMO),
    the intersection is simply the overlapping region of the two arrays.

    Parameters
    ----------
    racmo : xarray.DataArray
        RACMO data array (2D) in EPSG:3031 projection.
    era5 : xarray.DataArray
        ERA5 data array (2D) in EPSG:3031 projection.

    Returns
    -------
    tuple
        A tuple containing the clipped RACMO and ERA5 data arrays.
    """
    # ── Intersection bounding box (RACMO ⊂ ERA5 here) ────────────────────────────
    era5_bounds  = era5.rio.bounds()    # (xmin, ymin, xmax, ymax)
    racmo_bounds = racmo.rio.bounds()

    # get bounding box 
    xmin = max(era5_bounds[0], racmo_bounds[0])
    ymin = max(era5_bounds[1], racmo_bounds[1])
    xmax = min(era5_bounds[2], racmo_bounds[2])
    ymax = min(era5_bounds[3], racmo_bounds[3])

    # ── Clip both datasets to intersection ───────────────────────────────────────
    racmo_clip = racmo.rio.clip_box(xmin, ymin, xmax, ymax)
    era5_clip  = era5.rio.clip_box(xmin, ymin, xmax, ymax)

    return racmo_clip, era5_clip


def plotTiles(era5: xr.DataArray, racmo: xr.DataArray, tiles: list, N: int = 30) -> plt.Figure:
    """
    Plot a subset of tiles with RACMO and ERA5 patches.

    Parameters
    ----------
    era5 : xarray.DataArray
        ERA5 data array (2D) in EPSG:3031 projection.
    racmo : xarray.DataArray
        RACMO data array (2D) in EPSG:3031 projection.
    tiles : list of dicts
        Each dict contains slices for RACMO and ERA5 tiles.
    N : int
        Number of tiles to plot (randomly sampled).
    """
    fig, ax = plt.subplots(figsize=(5,5),
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})

    racmo_x_vals = racmo.x.values
    racmo_y_vals = racmo.y.values

    ax.coastlines('50m', linewidth=0.5)
    ax.set_title("Subset of RACMO and ERA5 tiles")
    ax.set_extent([racmo_x_vals[0], racmo_x_vals[-1],
                    racmo_y_vals[-1], racmo_y_vals[0]],
                    crs=ccrs.SouthPolarStereo())

    subset_tiles = np.random.choice(tiles, size=min(N, len(tiles)), replace=False)
    for t in subset_tiles:   # draw a subset to keep the plot readable
        # RACMO tile rectangle (in projected metres)
        ry = t["racmo_slice_y"]
        rx = t["racmo_slice_x"]
        x0_r, x1_r = racmo_x_vals[rx.start], racmo_x_vals[rx.stop - 1]
        y0_r, y1_r = racmo_y_vals[ry.start], racmo_y_vals[ry.stop - 1]
        rect_r = mpatches.Rectangle((min(x0_r, x1_r), min(y0_r, y1_r)),
                                    abs(x1_r - x0_r), abs(y1_r - y0_r),
                                    linewidth=0.8, edgecolor='steelblue',
                                    facecolor='steelblue', alpha=0.3,
                                    transform=ccrs.SouthPolarStereo())
        ax.add_patch(rect_r)

        # ERA5 tile rectangle
        ey = t["era5_slice_y"]
        ex = t["era5_slice_x"]
        era5_x_vals = era5.x.values
        era5_y_vals = era5.y.values
        x0_e, x1_e = era5_x_vals[ex.start], era5_x_vals[ex.stop - 1]
        y0_e, y1_e = era5_y_vals[ey.start], era5_y_vals[ey.stop - 1]
        rect_e = mpatches.Rectangle((min(x0_e, x1_e), min(y0_e, y1_e)),
                                    abs(x1_e - x0_e), abs(y1_e - y0_e),
                                    linewidth=0.8, edgecolor='coral',
                                    facecolor='coral', alpha=0.3,
                                    transform=ccrs.SouthPolarStereo())
        ax.add_patch(rect_e)
    
    plt.tight_layout()
    plt.close()
    return fig


def slice_to_dict(slice_obj: slice) -> dict:
    """
    Convert a slice object to a dictionary representation.

    Parameters
    ----------
    slice_obj : slice
        A slice object to convert.

    Returns
    -------
    dict
        A dictionary with 'start', 'stop', and 'step' keys.
    """
    return {"start": slice_obj.start,
            "stop": slice_obj.stop,
            "step": slice_obj.step}


def dict_to_slice(slice_dict: dict) -> slice:
    """
    Convert a dictionary representation of a slice back to a slice object.

    Parameters
    ----------
    slice_dict : dict
        A dictionary with 'start', 'stop', and 'step' keys.

    Returns
    -------
    slice
        A slice object.
    """
    return slice(slice_dict["start"], slice_dict["stop"], slice_dict["step"])


def save_tiles_metadata(tiles: list, filename: str) -> None:
    """
    Save tile metadata to a JSON file.

    Parameters
    ----------
    tiles : list of dicts
        Each dict contains slices for RACMO and ERA5 tiles.
    filename : str
        Path to the output JSON file.
    """
    tiles_id = np.arange(1, len(tiles)+1)
    tiles_id = [str(i) for i in tiles_id]  # convert to strings for better display

    id_tile_dict = dict(zip(tiles_id, tiles))

    json_dict = {}
    for k, v in id_tile_dict.items():
        json_dict[k] = {"racmo_slice_y": slice_to_dict(v["racmo_slice_y"]),
                        "racmo_slice_x": slice_to_dict(v["racmo_slice_x"]),
                        "era5_slice_y": slice_to_dict(v["era5_slice_y"]),
                        "era5_slice_x": slice_to_dict(v["era5_slice_x"])}
        
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=4)


def load_tiles_metadata(filename: str) -> list:
    """
    Load tile metadata from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the input JSON file.

    Returns
    -------
    list of dicts
        Each dict contains slices for RACMO and ERA5 tiles.
    """
    with open(filename, 'r') as f:
        json_dict = json.load(f)

    tiles = []
    for k, v in json_dict.items():
        tiles.append({"racmo_slice_y": dict_to_slice(v["racmo_slice_y"]),
                      "racmo_slice_x": dict_to_slice(v["racmo_slice_x"]),
                      "era5_slice_y": dict_to_slice(v["era5_slice_y"]),
                      "era5_slice_x": dict_to_slice(v["era5_slice_x"])})
    
    return tiles
    