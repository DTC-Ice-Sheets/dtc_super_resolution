"""Tests for the RCM emulator entrypoint."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_parse_args_valid(tmpdir):
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--start_date",
            "2021-01-01",
            "--end_date",
            "2021-06-01",
            "--output_dir",
            str(tmpdir / "out"),
            "--model_name",
            "model_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import parse_args

        args = parse_args()

    assert args.start_date == "2021-01-01"
    assert args.end_date == "2021-06-01"
    assert args.output_dir == str(tmpdir / "out")
    assert args.model_name == "model_unet"


def test_parse_args_invalid_model_choice(tmpdir):
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--start_date",
            "2021-01-01",
            "--end_date",
            "2021-06-01",
            "--output_dir",
            str(tmpdir / "out"),
            "--model_name",
            "invalid_model",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import parse_args

        with pytest.raises(SystemExit):
            parse_args()


def test_parse_args_attention_unet(tmpdir):
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--start_date",
            "2020-01-01",
            "--end_date",
            "2020-12-01",
            "--output_dir",
            str(tmpdir / "out"),
            "--model_name",
            "model_attention_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import parse_args

        args = parse_args()

    assert args.model_name == "model_attention_unet"


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
            "--start_date",
            "2021-01-01",
            "--end_date",
            "2021-06-01",
            "--output_dir",
            str(tmp_path / "new_dir"),
            "--model_name",
            "model_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import main

        main()

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
            "--start_date",
            "2021-01-01",
            "--end_date",
            "2021-05-01",
            "--output_dir",
            str(tmpdir / "out"),
            "--model_name",
            "model_unet",
        ],
    ):
        from super_resolution.rcm_emulator_entrypoint import main

        main()

    mock_preproc.assert_called_once()
    mock_infer.assert_called_once_with("model_unet", mock_patches, mock_metadata)
    mock_postproc.assert_called_once_with(mock_monthly, mock_metadata, mock_tiles, mock_rcm_ds, mock_gcm_ds)
    mock_final_da.to_zarr.assert_called_once()
    mock_plot.assert_called_once()
