"""Tests for the download_data module."""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, call, patch
from super_resolution.download_data import download_era5_from_edh, download_rcm_from_ddm, download_data

import pytest
from typeguard import suppress_type_checks

def make_fake_era5_dataset():
    lats = np.linspace(-40, -90, 10)  # descending, matching real ERA5 latitude ordering
    lons = np.linspace(0, 360, 20)
    times = pd.date_range("2010-01-01", "2023-12-31", freq="MS")
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.Dataset(
        {"t2m": (["valid_time", "latitude", "longitude"], data)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )


def test_download_era5_from_edh_saves_netcdf(tmp_path):
    """The function returns a path string (annotation says xr.Dataset — suppress typeguard)."""
    fake_ds = make_fake_era5_dataset()
    with patch("super_resolution.download_data.xr.open_dataset", return_value=fake_ds) as mock_open:
        with patch.object(xr.Dataset, "to_netcdf") as mock_to_netcdf:
            result_path = download_era5_from_edh(
                start_date="2010-01-01",
                end_date="2010-12-31",
                output_dir=str(tmp_path),
            )

    mock_open.assert_called_once()
    called_url = mock_open.call_args[0][0]
    assert "reanalysis-era5-single-levels-monthly-means-v0.zarr" in called_url

    mock_to_netcdf.assert_called_once()
    assert result_path == str(tmp_path / "ERA5_monthly_means_2010-01-01_2010-12-31.nc")


def test_slices_to_antarctica_region(tmp_path):
    fake_ds = make_fake_era5_dataset()
    captured = {}

    def fake_to_netcdf(self, path):
        captured["ds"] = self

    with patch("super_resolution.download_data.xr.open_dataset", return_value=fake_ds):
        with patch.object(xr.Dataset, "to_netcdf", fake_to_netcdf):
            download_era5_from_edh("2010-01-01", "2010-12-31", str(tmp_path))

    sliced = captured["ds"]
    assert sliced.latitude.max() <= -40
    assert sliced.latitude.min() >= -90
    assert sliced.valid_time.min() >= np.datetime64("2010-01-01")
    assert sliced.valid_time.max() <= np.datetime64("2010-12-31")


@patch("super_resolution.download_data.ddm.main.download_data")
@patch("super_resolution.download_data.ddm.schemas.DataDownloadRequest")
def test_download_rcm_from_ddm(mock_request_cls, mock_download, tmp_path):
    mock_download.return_value = "/some/path.zarr"

    from super_resolution.download_data import download_rcm_from_ddm

    with suppress_type_checks():
        result = download_rcm_from_ddm(str(tmp_path), "2020-01-01", "2020-02-01")

    assert result == "/some/path.zarr"
    mock_request_cls.assert_called_once()
    mock_download.assert_called_once()


@patch("super_resolution.download_data.ddm.main.download_data")
@patch("super_resolution.download_data.ddm.schemas.DataDownloadRequest")
def test_download_rcm_from_ddm_correct_dates(mock_request_cls, mock_download, tmp_path):
    mock_download.return_value = "/some/path.zarr"

    from super_resolution.download_data import download_rcm_from_ddm

    with suppress_type_checks():
        download_rcm_from_ddm(str(tmp_path), "2020-03-01", "2020-04-01")

    call_kwargs = mock_request_cls.call_args[1]
    assert call_kwargs["start_time"] == datetime(2020, 3, 1)
    assert call_kwargs["end_time"] == datetime(2020, 4, 1)


@patch("super_resolution.download_data.download_rcm_from_ddm", return_value="/rcm/path.zarr")
@patch("super_resolution.download_data.download_era5_from_edh", return_value="/era5/path.nc")
@patch("super_resolution.download_data.dotenv.load_dotenv")
def test_download_data_returns_dict(mock_dotenv, mock_era5, mock_rcm, tmp_path):
    from super_resolution.download_data import download_data

    result = download_data("2021-01-01", "2021-06-01", str(tmp_path))

    assert result == {"era5": "/era5/path.nc", "rcm": "/rcm/path.zarr"}
    mock_era5.assert_called_once_with("2021-01-01", "2021-06-01", str(tmp_path))
    mock_rcm.assert_called_once_with(str(tmp_path))
