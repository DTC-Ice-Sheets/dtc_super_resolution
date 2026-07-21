"""Tests for inference_helpers module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rioxarray  # registers the .rio accessor on xr.DataArray
import torch
import xarray as xr
from typeguard import suppress_type_checks


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

@patch("super_resolution.ml_module.inference_helpers.torch.load")
@patch("super_resolution.ml_module.inference_helpers.UNetBaseline")
def test_load_model_baseline(mock_unet_cls, mock_torch_load):
    mock_model = MagicMock()
    mock_unet_cls.return_value = mock_model
    mock_torch_load.return_value = {}

    from super_resolution.ml_module.inference_helpers import load_model

    config = {
        "model_type": "Baseline",
        "n_channel_x": 7,
        "n_channel_z": 16,
        "size": 64,
        "filter": 64,
        "weights_path": "model_weights/best_model_attention.pth",
    }

    result = load_model(config)

    mock_unet_cls.assert_called_once_with(
        n_channels_x=7, n_channels_z=16, size=64, filter=64, bilinear=False
    )
    mock_model.load_state_dict.assert_called_once()
    assert result is mock_model


@patch("super_resolution.ml_module.inference_helpers.torch.load")
@patch("super_resolution.ml_module.inference_helpers.SmaAt_UNet")
def test_load_model_attention(mock_attn_cls, mock_torch_load):
    mock_model = MagicMock()
    mock_attn_cls.return_value = mock_model
    mock_torch_load.return_value = {}

    from super_resolution.ml_module.inference_helpers import load_model

    config = {
        "model_type": "Attention",
        "n_channel_x": 7,
        "n_channel_z": 16,
        "weights_path": "model_weights/best_model_attention.pth",
    }

    result = load_model(config)

    mock_attn_cls.assert_called_once_with(
        n_channels_x=7, n_channels_z=16, bilinear=False
    )
    assert result is mock_model


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def _make_input_patches(n_months=2, n_tiles=3, c=8, h=6, w=8):
    patches = np.random.rand(n_months, n_tiles, c, h, w).astype(np.float32)
    patches[:, :, 3, :, :] = 0.0  # SST = 0 (land)
    return patches


def _make_metadata(n_months=2, n_tiles=3):
    return [
        [
            {"time_idx": np.datetime64(f"2021-0{m+1}-01", "D"), "tile_idx": t}
            for t in range(n_tiles)
        ]
        for m in range(n_months)
    ]


def test_predict_returns_array():
    from super_resolution.ml_module.inference_helpers import predict

    mock_model = MagicMock()
    # Output shape: (1, 1, H, W) — mimicking model output for one tile
    mock_model.return_value = torch.zeros(1, 1, 6, 8)
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)

    input_patches = _make_input_patches(n_months=1, n_tiles=2)
    metadata = _make_metadata(n_months=1, n_tiles=2)

    config = {"target_transform": None}
    device = torch.device("cpu")

    with suppress_type_checks():
        result = predict(mock_model, input_patches, metadata, device=device, config=config)

    assert isinstance(result, np.ndarray)


def test_predict_uses_cpu_when_no_device():
    from super_resolution.ml_module.inference_helpers import predict

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 1, 4, 4)
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)

    input_patches = _make_input_patches(n_months=1, n_tiles=1, h=4, w=4)
    metadata = _make_metadata(n_months=1, n_tiles=1)

    with suppress_type_checks():
        predict(mock_model, input_patches, metadata, device=None, config={})

    mock_model.to.assert_called()
    mock_model.eval.assert_called()


# ---------------------------------------------------------------------------
# run_postprocessing
# ---------------------------------------------------------------------------

def test_run_postprocessing_output_shape():
    from super_resolution.ml_module.inference_helpers import run_postprocessing

    H, W = 20, 30
    n_months = 2
    n_tiles = 3

    tiles = [
        {"racmo_slice_y": slice(0, 10), "racmo_slice_x": slice(0, 15)},
        {"racmo_slice_y": slice(5, 15), "racmo_slice_x": slice(10, 25)},
        {"racmo_slice_y": slice(10, 20), "racmo_slice_x": slice(15, 30)},
    ]

    monthly_predictions = np.random.rand(n_months, n_tiles, 10, 15).astype(np.float32)

    metadata = [
        [{"time_idx": np.datetime64(f"2021-0{m+1}-01", "D"), "tile_idx": t} for t in range(n_tiles)]
        for m in range(n_months)
    ]

    rcm_ds = MagicMock()
    rcm_ds.rio.shape = (H, W)
    rcm_ds.y.values = np.arange(H, dtype=float)
    rcm_ds.x.values = np.arange(W, dtype=float)

    era5_xy = MagicMock()
    era5_xy.valid_time.values = np.array(
        ["2021-01-01", "2021-02-01"], dtype="datetime64[D]"
    )

    with suppress_type_checks():
        result = run_postprocessing(monthly_predictions, metadata, tiles, rcm_ds, era5_xy)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("valid_time", "y", "x")
    assert result.shape == (n_months, H, W)


def test_run_postprocessing_averages_overlapping_tiles():
    from super_resolution.ml_module.inference_helpers import run_postprocessing

    H, W = 10, 10
    # Two tiles that perfectly overlap the same region
    tiles = [
        {"racmo_slice_y": slice(0, 5), "racmo_slice_x": slice(0, 5)},
        {"racmo_slice_y": slice(0, 5), "racmo_slice_x": slice(0, 5)},
    ]
    monthly_predictions = np.array(
        [[np.ones((5, 5)), np.full((5, 5), 3.0)]], dtype=np.float32
    )
    metadata = [[
        {"time_idx": np.datetime64("2021-01-01", "D"), "tile_idx": 0},
        {"time_idx": np.datetime64("2021-01-01", "D"), "tile_idx": 1},
    ]]
    rcm_ds = MagicMock()
    rcm_ds.rio.shape = (H, W)
    rcm_ds.y.values = np.arange(H, dtype=float)
    rcm_ds.x.values = np.arange(W, dtype=float)
    era5_xy = MagicMock()
    era5_xy.valid_time.values = np.array(["2021-01-01"], dtype="datetime64[D]")

    with suppress_type_checks():
        result = run_postprocessing(monthly_predictions, metadata, tiles, rcm_ds, era5_xy)

    # Expected average of 1.0 and 3.0 = 2.0
    np.testing.assert_allclose(result.values[0, :5, :5], 2.0, atol=1e-5)
