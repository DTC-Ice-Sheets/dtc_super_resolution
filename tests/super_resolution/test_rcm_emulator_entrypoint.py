"""Tests for the RCM emulator entrypoint."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_app_valid_args(tmp_path):
    with (
        patch(
            "super_resolution.rcm_emulator_entrypoint.run_preprocessing",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch("super_resolution.rcm_emulator_entrypoint.run_inference", return_value=MagicMock()),
        patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing", return_value=MagicMock()),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(
            sys,
            "argv",
            [
                "prog",
                "--gcm_zarr_path",
                "era5.zarr",
                "--rcm_zarr_path",
                "rcm.zarr",
                "--output_zarr_path",
                str(tmp_path),
                "--model-name",
                "model_baseline_unet",
            ],
        ),
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")


def test_app_invalid_model_choice(tmp_path):
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--gcm_zarr_path",
            "era5.zarr",
            "--rcm_zarr_path",
            "rcm.zarr",
            "--output_zarr_path",
            str(tmp_path),
            "--model-name",
            "invalid_model",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        with pytest.raises(SystemExit) as exc_info:
            app(result_action="return_value")
        assert "1" in str(exc_info.value)


def test_app_attention_unet(tmp_path):
    with (
        patch(
            "super_resolution.rcm_emulator_entrypoint.run_preprocessing",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch("super_resolution.rcm_emulator_entrypoint.run_inference", return_value=MagicMock()) as mock_run_inference,
        patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing", return_value=MagicMock()),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(
            sys,
            "argv",
            [
                "prog",
                "--gcm_zarr_path",
                "era5.zarr",
                "--rcm_zarr_path",
                "rcm.zarr",
                "--output_zarr_path",
                str(tmp_path),
                "--model-name",
                "model_attention_unet",
            ],
        ),
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")
        called_model_name = mock_run_inference.call_args[0][0]
        assert called_model_name == "model_attention_unet"


@patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing")
@patch("super_resolution.rcm_emulator_entrypoint.run_inference")
@patch("super_resolution.rcm_emulator_entrypoint.run_preprocessing")
@patch("pathlib.Path.exists", new=MagicMock(return_value=True))
def test_main_calls_pipeline_stages(mock_preproc, mock_infer, mock_postproc, tmpdir):
    mock_patches = MagicMock()
    mock_metadata = MagicMock()
    mock_tiles = MagicMock()
    mock_rcm_ds = MagicMock()
    mock_gcm_ds = MagicMock()
    mock_preproc.return_value = (mock_patches, mock_metadata, mock_tiles, mock_rcm_ds, mock_gcm_ds)
    mock_monthly = MagicMock()
    mock_infer.return_value = mock_monthly
    mock_final_da = MagicMock()
    mock_postproc.return_value = mock_final_da

    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--gcm_zarr_path",
            "era5.zarr",
            "--rcm_zarr_path",
            "rcm.zarr",
            "--output_zarr_path",
            str(tmpdir / "out"),
            "--model-name",
            "model_baseline_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")

    mock_preproc.assert_called_once()
    mock_infer.assert_called_once_with("model_baseline_unet", mock_patches, mock_metadata)
    mock_postproc.assert_called_once_with(mock_monthly, mock_metadata, mock_tiles, mock_rcm_ds, mock_gcm_ds)
    mock_final_da.to_zarr.assert_called_once()
