"""Tests for the reprojection module."""

from unittest.mock import MagicMock, patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from typeguard import suppress_type_checks


def _make_racmo_ds(nx=20, ny=15):
    """Create a minimal RACMO-like Dataset in EPSG:3031."""
    x = np.arange(nx, dtype=float) * 2000.0
    y = np.arange(ny, dtype=float) * 2000.0
    data = np.ones((1, ny, nx), dtype=np.float32)
    ds = xr.Dataset(
        {"surface_mass_balance_glaciated": (["time", "y", "x"], data)},
        coords={"x": x, "y": y, "time": [0]},
    )
    return ds.rio.write_crs("epsg:3031")


def test_create_target_grid_from_racmo_within_bounds():
    from super_resolution.rcm_emulator.pre_processing_module.reprojection import create_target_grid_from_racmo

    racmo_ds = _make_racmo_ds(nx=40, ny=30)
    gridx, gridy = create_target_grid_from_racmo(racmo_ds, resx_km=14.0, resy_km=28.0)

    x_orig = racmo_ds.x.values
    y_orig = racmo_ds.y.values

    assert float(gridx.min()) >= float(x_orig.min())
    assert float(gridx.max()) <= float(x_orig.max())
    assert float(gridy.min()) >= float(y_orig.min())
    assert float(gridy.max()) <= float(y_orig.max())


def test_create_target_grid_from_racmo_coarser_resolution():
    from super_resolution.rcm_emulator.pre_processing_module.reprojection import create_target_grid_from_racmo

    racmo_ds = _make_racmo_ds(nx=60, ny=40)
    gridx, gridy = create_target_grid_from_racmo(racmo_ds, resx_km=14.0, resy_km=28.0)

    # Coarser grid should have fewer points than RACMO
    assert len(gridx) < len(racmo_ds.x)
    assert len(gridy) < len(racmo_ds.y)


def test_create_target_grid_returns_different_resolutions():
    from super_resolution.rcm_emulator.pre_processing_module.reprojection import create_target_grid_from_racmo

    racmo_ds = _make_racmo_ds(nx=60, ny=60)
    gridx_fine, _ = create_target_grid_from_racmo(racmo_ds, resx_km=2.0, resy_km=2.0)
    gridx_coarse, _ = create_target_grid_from_racmo(racmo_ds, resx_km=14.0, resy_km=28.0)

    assert len(gridx_fine) >= len(gridx_coarse)


@patch("super_resolution.pre_processing_module.reprojection.pyproj.Transformer.from_crs")
def test_reproject_era5_to_racmo_grid_calls_transformer(mock_from_crs):
    from super_resolution.rcm_emulator.pre_processing_module.reprojection import reproject_era5_to_racmo_grid

    # ERA5 uses 0-360 longitude; avoid 0 and 360 together (they normalize to same value)
    lats = np.linspace(-90, -60, 6)
    lons = np.linspace(0, 350, 8)  # 0 to 350 (not 360) to prevent duplicate after normalization
    times = np.array(["2021-01-01"], dtype="datetime64[D]")
    data = np.random.rand(1, len(lats), len(lons)).astype(np.float32)
    era5 = xr.Dataset(
        {"t2m": (["time", "y", "x"], data)},
        coords={"y": lats, "x": lons, "time": times},
    )

    gridx = xr.DataArray(np.array([100000.0, 200000.0, 300000.0]), dims="x")
    gridy = xr.DataArray(np.array([100000.0, 200000.0]), dims="y")

    # Mock the transformer to return lon/lat values within ERA5 range
    mock_transformer = MagicMock()
    mock_from_crs.return_value = mock_transformer
    xx, yy = np.meshgrid(gridx.values, gridy.values)
    mock_transformer.transform.return_value = (
        np.full_like(xx, 10.0),  # fake lons within ERA5 range
        np.full_like(yy, -75.0),  # fake lats within ERA5 range
    )

    with suppress_type_checks():
        result = reproject_era5_to_racmo_grid(era5, gridx, gridy)

    mock_from_crs.assert_called_once()
    mock_transformer.transform.assert_called_once()
    assert "x" in result.coords
    assert "y" in result.coords


def test_reproject_era5_to_racmo_grid_output_shape():
    """Integration test: reproject a tiny ERA5 grid onto a small polar grid."""
    from super_resolution.rcm_emulator.pre_processing_module.reprojection import reproject_era5_to_racmo_grid

    # ERA5 uses 0-360 longitude; avoid 0 and 360 together (they normalize to same value)
    lats = np.linspace(-90, -60, 15)
    lons = np.linspace(1, 359, 20)  # avoid 0 and 360 to prevent duplicate after normalization
    times = np.array(["2021-01-01"], dtype="datetime64[D]")
    data = np.random.rand(1, len(lats), len(lons)).astype(np.float32)
    era5 = xr.Dataset(
        {"t2m": (["time", "y", "x"], data)},
        coords={"y": lats, "x": lons, "time": times},
    )

    gridx = xr.DataArray(np.array([-500000.0, 0.0, 500000.0]), dims="x")
    gridy = xr.DataArray(np.array([-500000.0, 0.0, 500000.0]), dims="y")

    with suppress_type_checks():
        result = reproject_era5_to_racmo_grid(era5, gridx, gridy)

    assert result.dims["x"] == 3
    assert result.dims["y"] == 3
    assert "t2m" in result.data_vars
