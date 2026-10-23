"""Functions needed for using the trained model on new data."""

import logging
import pathlib

import numpy as np
import rioxarray  # noqa: F401
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_unet import SmaatUnet
from super_resolution.rcm_emulator.ml_module.basic_unet.unet import UNetBaseline
from super_resolution.rcm_emulator.ml_module.datasets import InferenceDataset

# from super_resolution.monitoring import setup_logging
# setup_logging()
logger = logging.getLogger(__name__)


def load_model(config: dict) -> nn.Module:
    """
    Load a trained model from the specified path.

    Parameters
    ----------
    config: dict
        A dictionary containing the model configuration, including the model type and weights path.

    Returns
    -------
    nn.Module
        The loaded PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model_type"] == "Baseline":
        model = UNetBaseline(
            n_channels_x=config["n_channel_x"],
            n_channels_z=config["n_channel_z"],
            size=config["size"],
            n_filter=config["filter"],
            bilinear=False,
        )

    elif config["model_type"] == "Attention":
        model = SmaatUnet(n_channels_x=config["n_channel_x"], n_channels_z=config["n_channel_z"], bilinear=False)

    # load the model weights, supporting both remote URLs and local paths
    weights_path = config["weights_path"]
    if str(weights_path).startswith("http://") or str(weights_path).startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(weights_path, weights_only=False, map_location=device)
    else:
        model_path = pathlib.Path.cwd() / weights_path
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    return model


def predict(model: nn.Module, input_patches: np.ndarray, metadata: list, device: torch.device) -> dict:
    """
    Run inference on the model using the provided DataLoader.

    Parameters
    ----------
    model: nn.Module
        The trained PyTorch model.
    input_patches: np.ndarray
        A list of numpy arrays containing preprocessed input patches for inference.
    metadata: list
        A list of dictionaries containing metadata for each input patch.
    device: torch.device
        The device to run the inference on (CPU or GPU).

    Returns
    -------
        A dictionary containing the model's predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # set the model to evaluation mode
    model.to(device)
    monthly_predictions = []
    # first dimension of input_patches is the number of months
    # (months, tiles, channels, height, width)

    for m in range(input_patches.shape[0]):
        predictions = []
        dataset = InferenceDataset(input_patches[m], metadata[m])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # predict all the tiles for this month
        for x, z in tqdm(dataloader, desc=f"Month {m + 1}/{input_patches.shape[0]}"):
            x = x.to(device)
            z = z.to(device)
            with torch.no_grad():
                out = model(x, z)

                predictions.append(out.cpu().numpy())
        # for full antartica tiles = 139, for 1 location, tiles = 1
        monthly_predictions.append(predictions)  # (months, tiles, height, width)

    return np.array(monthly_predictions)    # always (T, N, H, W), also for T=1


def run_postprocessing(
    monthly_predictions: np.ndarray, metadata: list, tiles: list, rcm_ds: xr.Dataset, era5_xy: xr.Dataset
) -> np.ndarray:
    """
    Post-process the model's predictions to match the original input shape.

    Parameters
    ----------
    monthly_predictions: np.ndarray
        The model's predictions.
    metadata: list
        A list of dictionaries containing metadata for each input patch.
    tiles: list
        A list of dictionaries containing metadata for each tile.
    rcm_ds: xr.Dataset
        The RCM dataset.
    era5_xy: xr.Dataset
        The ERA5 dataset reprojected to the RCM grid.

    Returns
    -------
    np.ndarray
        The post-processed predictions.
    """
    # reshape predictions to match the original input shape
    # (months, tiles, height, width)
    reshaped_predictions = []

    # Build a 2-D Gaussian weight map matching the tile output shape.
    # Weights peak at the tile centre and decay toward the edges, so that
    # edge predictions (lower model confidence due to limited receptive field)
    # contribute less than centre predictions when tiles overlap.
    tile_h, tile_w = monthly_predictions.shape[2], monthly_predictions.shape[3]
    cy, cx = (tile_h - 1) / 2.0, (tile_w - 1) / 2.0
    sigma_y, sigma_x = tile_h / 4.0, tile_w / 4.0
    yy, xx = np.ogrid[:tile_h, :tile_w]
    gaussian_weights = np.exp(-((yy - cy) ** 2 / (2 * sigma_y**2) + (xx - cx) ** 2 / (2 * sigma_x**2))).astype(
        np.float32
    )

    # populate with predictions
    for i in range(monthly_predictions.shape[0]):
        # full racmo grid
        rcm_preds_acc = np.zeros(rcm_ds.rio.shape)
        rcm_preds_w = np.zeros(rcm_ds.rio.shape)

        for j in range(monthly_predictions.shape[1]):
            meta = metadata[i][j]
            pred = monthly_predictions[i, j, :, :]
            tile_id = meta["tile_idx"]
            tile = tiles[tile_id]

            ry, rx = tile["racmo_slice_y"], tile["racmo_slice_x"]
            h = ry.stop - ry.start
            w = rx.stop - rx.start
            # support partial tiles at the domain boundary
            w_crop = gaussian_weights[:h, :w]
            rcm_preds_acc[ry, rx] += pred[:h, :w] * w_crop
            rcm_preds_w[ry, rx] += w_crop

        # compute the final predictions
        with np.errstate(invalid="ignore"):
            rcm_preds = np.where(rcm_preds_w > 0, rcm_preds_acc / rcm_preds_w, np.nan)
        reshaped_predictions.append(rcm_preds)

    reshaped_predictions = np.array(reshaped_predictions)  # (T, H, W)

    return xr.DataArray(
        data=reshaped_predictions,
        dims=["time", "y", "x"],
        coords={"time": era5_xy.time.values, "y": rcm_ds.y.values, "x": rcm_ds.x.values},
        name="surface_mass_balance_glaciated",
    ).rio.write_crs("epsg:3031")


def run_postprocessing_loc(
    predictions: np.ndarray,
    tile_meta: list,
    rcm_ds: xr.Dataset,
    era5_xy: xr.Dataset,
    input_patches: np.ndarray,
    target_patches: np.ndarray,
) -> np.ndarray:
    """
    Post-process the model's predictions to match the original input shape for a specific location.

    Parameters
    ----------
    predictions: np.ndarray
        The model's predictions.
    tile_meta: list
        A list of dictionaries containing metadata for each input patch.
    rcm_ds: xr.Dataset
        The RCM dataset.
    era5_xy: xr.Dataset
        The ERA5 dataset reprojected to the RCM grid.
    input_patches: np.ndarray
        The input patches extracted from the ERA5 dataset.
    target_patches: np.ndarray
        The target patches extracted from the RCM dataset.

    Returns
    -------
    np.ndarray
        The post-processed predictions for the specific location.
    """
    if predictions.shape[0] == 1:
        predictions = predictions.reshape(1, predictions.shape[-2], predictions.shape[-1])  # (1, H, W)
    else:
        predictions = np.squeeze(predictions)  # (time, y, x)

    # Build a serializable copy of tile_meta (convert slice objects to dicts)
    tile_meta_serializable = {
        "lon": float(tile_meta["lon"]),
        "lat": float(tile_meta["lat"]),
        "x_3031": float(tile_meta["x_3031"]),
        "y_3031": float(tile_meta["y_3031"]),
        "era5_y_slice": {
            "start": int(tile_meta["era5_y_slice"].start),
            "stop": int(tile_meta["era5_y_slice"].stop),
            "step": tile_meta["era5_y_slice"].step,
        },
        "era5_x_slice": {
            "start": int(tile_meta["era5_x_slice"].start),
            "stop": int(tile_meta["era5_x_slice"].stop),
            "step": tile_meta["era5_x_slice"].step,
        },
        "racmo_y_slice": {
            "start": int(tile_meta["racmo_y_slice"].start),
            "stop": int(tile_meta["racmo_y_slice"].stop),
            "step": tile_meta["racmo_y_slice"].step,
        },
        "racmo_x_slice": {
            "start": int(tile_meta["racmo_x_slice"].start),
            "stop": int(tile_meta["racmo_x_slice"].stop),
            "step": tile_meta["racmo_x_slice"].step,
        },
    }

    # racmo grid
    ry = tile_meta["racmo_y_slice"]
    rx = tile_meta["racmo_x_slice"]

    racmo_x_vals = rcm_ds.x.values
    racmo_y_vals = rcm_ds.y.values

    # 1-D coordinate arrays for the patches
    rx_coords = racmo_x_vals[rx]
    ry_coords = racmo_y_vals[ry]

    # era5 grid
    ry_era5 = tile_meta["era5_y_slice"]
    rx_era5 = tile_meta["era5_x_slice"]
    era5_x_vals = era5_xy.x.values
    era5_y_vals = era5_xy.y.values
    ex_coords = era5_x_vals[rx_era5]
    ey_coords = era5_y_vals[ry_era5]
    context_pad_h = 16  # extra ERA5 pixels on top/bottom
    context_pad_w = 8  # extra ERA5 pixels on left/right
    ex_coords_cropped = ex_coords[context_pad_w : len(ex_coords) - context_pad_w]
    ey_coords_cropped = ey_coords[context_pad_h : len(ey_coords) - context_pad_h]

    # drop the single-tile dimension: (T, 1, C, H, W) -> (T, C, H, W), (T, 1, H, W) -> (T, H, W)
    input_patches = np.squeeze(input_patches, axis=1)
    target_patches = np.squeeze(target_patches, axis=1)

    x_cropped = input_patches[
        :,
        :,
        context_pad_h : input_patches.shape[2] - context_pad_h,
        context_pad_w : input_patches.shape[3] - context_pad_w,
    ]

    # Create dataset
    return xr.Dataset(
        data_vars={
            "predictions": (("time", "y", "x"), predictions.astype(np.float32)),
            "target_patches": (("time_r", "y", "x"), target_patches.astype(np.float32)),
            "x_cropped": (("time", "channels", "y_era5", "x_era5"), x_cropped.astype(np.float32)),
        },
        coords={
            "time": era5_xy.time.values,
            "time_r": rcm_ds.time.values,
            "y": ry_coords,
            "x": rx_coords,
            "y_era5": ey_coords_cropped,
            "x_era5": ex_coords_cropped,
            "channels": ["d2m", "msl", "sp", "sst", "t2m", "tp", "u10", "v10"],
        },
        attrs=tile_meta_serializable,
    )
