"""Tests for the RCM emulator entrypoint."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_app_valid_args(tmp_path):
    with (
        patch("super_resolution.rcm_emulator_entrypoint.download_data", return_value=MagicMock()),
        patch(
            "super_resolution.rcm_emulator_entrypoint.run_preprocessing",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch("super_resolution.rcm_emulator_entrypoint.run_inference", return_value=MagicMock()),
        patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing", return_value=MagicMock()),
        patch("super_resolution.rcm_emulator_entrypoint.predictions_plot"),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(
            sys,
            "argv",
            [
                "prog",
                "--start-date",
                "2021-01-01",
                "--end-date",
                "2021-06-01",
                "--output-dir",
                str(tmp_path),
                "--model-name",
                "model_unet",
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
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-06-01",
            "--output-dir",
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
        patch("super_resolution.rcm_emulator_entrypoint.download_data", return_value=MagicMock()) as mock_download,
        patch(
            "super_resolution.rcm_emulator_entrypoint.run_preprocessing",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch("super_resolution.rcm_emulator_entrypoint.run_inference", return_value=MagicMock()) as mock_run_inference,
        patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing", return_value=MagicMock()),
        patch("super_resolution.rcm_emulator_entrypoint.predictions_plot"),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(
            sys,
            "argv",
            [
                "prog",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-01",
                "--output-dir",
                str(tmp_path),
                "--model-name",
                "model_attention_unet",
            ],
        ),
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")
        mock_download.assert_called_once_with("2020-01-01", "2020-12-01", str(tmp_path))
        called_model_name = mock_run_inference.call_args[0][0]
        assert called_model_name == "model_attention_unet"


@patch("super_resolution.rcm_emulator_entrypoint.predictions_plot", new=MagicMock())
@patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing")
@patch("super_resolution.rcm_emulator_entrypoint.run_inference")
@patch("super_resolution.rcm_emulator_entrypoint.run_preprocessing")
@patch("super_resolution.rcm_emulator_entrypoint.download_data", new=MagicMock())
@patch("pathlib.Path.exists", new=MagicMock(return_value=False))
@patch("pathlib.Path.mkdir")
def test_creates_dir_when_missing(mock_mkdir, mock_preproc, mock_infer, mock_postproc, tmp_path):
    mock_preproc.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_infer.return_value = MagicMock()
    mock_postproc.return_value = MagicMock()

    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-06-01",
            "--output-dir",
            str(tmp_path / "new_dir"),
            "--model-name",
            "model_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")

    mock_mkdir.assert_called_once_with(parents=True)


@patch("super_resolution.rcm_emulator_entrypoint.predictions_plot")
@patch("super_resolution.rcm_emulator_entrypoint.run_postprocessing")
@patch("super_resolution.rcm_emulator_entrypoint.run_inference")
@patch("super_resolution.rcm_emulator_entrypoint.run_preprocessing")
@patch("super_resolution.rcm_emulator_entrypoint.download_data", new=MagicMock())
@patch("pathlib.Path.exists", new=MagicMock(return_value=True))
def test_main_calls_pipeline_stages(mock_preproc, mock_infer, mock_postproc, mock_plot, tmpdir):
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
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-05-01",
            "--output-dir",
            str(tmpdir / "out"),
            "--model-name",
            "model_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import app

        app(result_action="return_value")

    mock_preproc.assert_called_once()
    mock_infer.assert_called_once_with("model_unet", mock_patches, mock_metadata)
    mock_postproc.assert_called_once_with(mock_monthly, mock_metadata, mock_tiles, mock_rcm_ds, mock_gcm_ds)
    mock_final_da.to_zarr.assert_called_once()
    mock_plot.assert_called_once()
