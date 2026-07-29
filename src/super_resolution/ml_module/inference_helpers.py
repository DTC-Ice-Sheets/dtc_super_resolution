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

from super_resolution.ml_module.attention_unet.rcm_emulator_unet import SmaAt_UNet
from super_resolution.ml_module.basic_unet.unet import UNetBaseline
from super_resolution.ml_module.datasets import InferenceDataset

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
    if config["model_type"] == "Baseline":
        model = UNetBaseline(
            n_channels_x=config["n_channel_x"],
            n_channels_z=config["n_channel_z"],
            size=config["size"],
            filter=config["filter"],
            bilinear=False,
        )

    elif config["model_type"] == "Attention":
        model = SmaAt_UNet(n_channels_x=config["n_channel_x"], n_channels_z=config["n_channel_z"], bilinear=False)

    # load the model weights
    model_path = pathlib.Path.cwd() / config["weights_path"]
    # model_path = Path(__file__).resolve().parent / config["weights_path"]
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model


def predict(model: nn.Module, input_patches: np.ndarray, metadata: list, device: torch.device, config: dict) -> dict:
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
    config: dict
        A dictionary containing the model configuration, including the model type and weights path.

    Returns
    -------
    dict
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
        dataset = InferenceDataset(input_patches[m], metadata[m], ownstd=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # predict all the tiles for this month
        for x, z in tqdm(dataloader, desc=f"Month {m + 1}/{input_patches.shape[0]}"):
            x = x.to(device)
            z = z.to(device)
            with torch.no_grad():
                out = model(x, z)
                # inverse-transform the predictions
                # try:
                #     if config["target_transform"] == "log":
                #         out = inverse_signed_log1p(out)
                #     elif config["target_transform"] == "normalization":
                #         out = out * dataset.target_train_std + dataset.target_train_mean
                # except AttributeError:
                #     logger.warning("Inverse transformation of targets not applied. Check if the model's output requires inverse transformation.")

                predictions.append(out.cpu().numpy())

        monthly_predictions.append(predictions)  # (12, 139, 224, 224)

    return np.squeeze(np.array(monthly_predictions))


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
    # (months, tiles, channels, height, width)
    reshaped_predictions = []
    # populate with predictions
    for i in range(monthly_predictions.shape[0]):
        # full racmo grid
        rcm_preds_acc = np.zeros(rcm_ds.rio.shape)
        rcm_preds_n = np.zeros(rcm_ds.rio.shape) + 1e-8

        for j in range(monthly_predictions.shape[1]):
            meta = metadata[i][j]
            pred = monthly_predictions[i, j, :, :]
            tile_id = meta["tile_idx"]
            tile = tiles[tile_id]

            ry, rx = tile["racmo_slice_y"], tile["racmo_slice_x"]
            rcm_preds_acc[ry, rx] += pred
            rcm_preds_n[ry, rx] += 1

        # compute the final predictions
        rcm_preds = rcm_preds_acc / rcm_preds_n
        mask = rcm_preds_n == 0.0
        rcm_preds[mask] = np.nan  # set to NaN where no predictions were made
        reshaped_predictions.append(rcm_preds)

    reshaped_predictions = np.array(reshaped_predictions)  # (T, H, W)

    return xr.DataArray(
        data=reshaped_predictions,
        dims=["time", "y", "x"],
        coords={"time": era5_xy.time.values, "y": rcm_ds.y.values, "x": rcm_ds.x.values},
        name="surface_mass_balance_glaciated",
    ).rio.write_crs("epsg:3031")
