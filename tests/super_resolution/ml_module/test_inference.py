"""Tests for the inference module."""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from typeguard import suppress_type_checks

MOCK_CONFIG = {
    "model_unet": {
        "model_type": "Baseline",
        "n_channel_x": 7,
        "n_channel_z": 16,
        "size": 64,
        "filter": 64,
        "weights_path": "model_weights/best_model.pth",
    },
    "model_attention_unet": {
        "model_type": "Attention",
        "n_channel_x": 7,
        "n_channel_z": 16,
        "weights_path": "model_weights/best_model_attention.pth",
    },
}


@patch("pathlib.Path.open", new=mock_open())
@patch("super_resolution.rcm_emulator.ml_module.inference.predict")
@patch("super_resolution.rcm_emulator.ml_module.inference.load_model")
@patch("super_resolution.rcm_emulator.ml_module.inference.yaml.safe_load", new=MagicMock(return_value=MOCK_CONFIG))
def test_run_inference_calls_load_model(mock_load, mock_predict):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    mock_predict.return_value = np.zeros((2, 3, 10, 10))

    from super_resolution.rcm_emulator.ml_module.inference import run_inference

    # run_inference expects list types (matching type annotations)
    input_patches = [[[0.0] * 3] * 3] * 2
    metadata = [[{"time_idx": 0}] * 3] * 2

    with suppress_type_checks():
        result = run_inference("model_unet", input_patches, metadata)

    mock_load.assert_called_once_with(MOCK_CONFIG["model_unet"])
    mock_predict.assert_called_once()
    assert isinstance(result, np.ndarray)


@patch("pathlib.Path.open", new=mock_open())
@patch("super_resolution.rcm_emulator.ml_module.inference.predict", new=MagicMock())
@patch("super_resolution.rcm_emulator.ml_module.inference.load_model", new=MagicMock())
@patch("super_resolution.rcm_emulator.ml_module.inference.yaml.safe_load", new=MagicMock(return_value=MOCK_CONFIG))
def test_run_inference_invalid_model_raises():
    from super_resolution.rcm_emulator.ml_module.inference import run_inference

    input_patches = [[0.0]]
    metadata = [[{"time_idx": 0}]]

    with suppress_type_checks(), pytest.raises(ValueError, match="not found in the configuration file"):
        run_inference("nonexistent_model", input_patches, metadata)


@patch("pathlib.Path.open", new=mock_open())
@patch("super_resolution.rcm_emulator.ml_module.inference.predict")
@patch("super_resolution.rcm_emulator.ml_module.inference.load_model")
@patch("super_resolution.rcm_emulator.ml_module.inference.yaml.safe_load", new=MagicMock(return_value=MOCK_CONFIG))
def test_run_inference_passes_correct_model_config(mock_load, mock_predict):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    mock_predict.return_value = np.zeros((1, 2, 8, 8))

    from super_resolution.rcm_emulator.ml_module.inference import run_inference

    input_patches = [[0.0]] * 2
    metadata = [[{"time_idx": 0}] * 2]

    with suppress_type_checks():
        run_inference("model_attention_unet", input_patches, metadata)

    mock_load.assert_called_once_with(MOCK_CONFIG["model_attention_unet"])
