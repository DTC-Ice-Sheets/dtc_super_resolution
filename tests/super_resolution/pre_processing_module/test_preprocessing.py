"""Tests for the preprocessing module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from typeguard import suppress_type_checks


@patch("super_resolution.pre_processing_module.preprocessing.xr.open_dataset")
def test_load_era5_dataset(mock_open):
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds

    from super_resolution.pre_processing_module.preprocessing import load_era5_dataset

    result = load_era5_dataset("/some/path.nc")

    mock_open.assert_called_once_with("/some/path.nc")
    assert result is mock_ds


@patch("super_resolution.pre_processing_module.preprocessing.xr.open_zarr")
def test_load_rcm_dataset(mock_open_zarr):
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.rio = MagicMock()
    mock_ds.rio.write_crs.return_value = mock_ds
    mock_open_zarr.return_value = mock_ds

    from super_resolution.pre_processing_module.preprocessing import load_rcm_dataset

    result = load_rcm_dataset("/some/rcm.zarr")

    mock_open_zarr.assert_called_once_with("/some/rcm.zarr", consolidated=True)
    mock_ds.rio.write_crs.assert_called_once_with("epsg:3031")
    assert result is mock_ds


def _make_era5_dataset(n_times=2, ny=10, nx=12):
    """Create a minimal ERA5-like xarray Dataset with spatial x/y dims."""
    times = np.array(["2021-01-01", "2021-02-01"], dtype="datetime64[D]")[:n_times]
    data = np.random.rand(n_times, ny, nx).astype(np.float32)
    ds = xr.Dataset(
        {
            "t2m": (["valid_time", "y", "x"], data),
            "u10": (["valid_time", "y", "x"], data * 2),
            "v10": (["valid_time", "y", "x"], data * 3),
            # SST channel at index 3 — set to zero to simulate "land" tiles
            "sst": (["valid_time", "y", "x"], np.zeros((n_times, ny, nx), dtype=np.float32)),
        },
        coords={
            "valid_time": times,
            "x": np.arange(nx, dtype=float),
            "y": np.arange(ny, dtype=float),
        },
    )
    return ds


def _make_tiles(era5_ny=10, era5_nx=12):
    """Return a small list of tile dicts compatible with _make_era5_dataset."""
    return [
        {
            "era5_slice_y": slice(0, 4),
            "era5_slice_x": slice(0, 6),
            "racmo_slice_y": slice(0, 8),
            "racmo_slice_x": slice(0, 12),
        },
        {
            "era5_slice_y": slice(4, 8),
            "era5_slice_x": slice(4, 10),
            "racmo_slice_y": slice(8, 16),
            "racmo_slice_x": slice(8, 20),
        },
    ]


def test_generate_era5_patches_shape():
    era5 = _make_era5_dataset(n_times=2, ny=10, nx=12)
    tiles = _make_tiles()

    from super_resolution.pre_processing_module.preprocessing import generate_era5_patches

    patches, metadata = generate_era5_patches(era5, tiles)

    # Shape: (T, N_kept, C, H, W) — N_kept may be 0 if SST is all zeros (land_mask.mean < 0.5)
    assert patches.ndim == 2 or patches.ndim == 5
    assert len(metadata) == 2  # one entry per time step


def test_generate_era5_patches_all_land_skipped():
    """Tiles with SST == 0 everywhere (pure land) are retained because land_mask.mean >= 0.5."""
    era5 = _make_era5_dataset(n_times=1, ny=10, nx=12)
    # Force SST to zero → land_mask.mean == 1.0 >= 0.5 → tiles should be KEPT
    era5["sst"].values[:] = 0.0
    tiles = _make_tiles()

    from super_resolution.pre_processing_module.preprocessing import generate_era5_patches

    patches, metadata = generate_era5_patches(era5, tiles)

    # Both tiles should be retained
    assert len(metadata[0]) == len(tiles)


def test_generate_era5_patches_sea_tiles_skipped():
    """Tiles with SST != 0 everywhere (pure ocean) are skipped."""
    era5 = _make_era5_dataset(n_times=1, ny=10, nx=12)
    # Force SST to non-zero → land_mask.mean == 0.0 < 0.5 → tiles should be SKIPPED
    era5["sst"].values[:] = 5.0
    tiles = _make_tiles()

    from super_resolution.pre_processing_module.preprocessing import generate_era5_patches

    patches, metadata = generate_era5_patches(era5, tiles)

    assert len(metadata[0]) == 0


@patch("super_resolution.pre_processing_module.preprocessing.generate_era5_patches")
@patch("super_resolution.pre_processing_module.preprocessing.make_tiles")
@patch("super_resolution.pre_processing_module.preprocessing.reproject_ERA5_to_RACMO_grid")
@patch("super_resolution.pre_processing_module.preprocessing.create_target_grid_from_RACMO")
@patch("super_resolution.pre_processing_module.preprocessing.load_rcm_dataset")
@patch("super_resolution.pre_processing_module.preprocessing.load_era5_dataset")
def test_run_preprocessing_calls_pipeline(
    mock_load_era5, mock_load_rcm, mock_create_grid,
    mock_reproject, mock_make_tiles, mock_gen_patches
):
    mock_gcm = MagicMock()
    mock_rcm = MagicMock()
    mock_gridx = MagicMock()
    mock_gridy = MagicMock()
    mock_era5_xy = MagicMock()
    mock_tiles = [MagicMock()]
    mock_patches = np.zeros((1, 1, 4, 4, 6))
    mock_metadata = [[{"time_idx": 0, "tile_idx": 0}]]

    mock_load_era5.return_value = mock_gcm
    mock_load_rcm.return_value = mock_rcm
    mock_create_grid.return_value = (mock_gridx, mock_gridy)
    mock_reproject.return_value = mock_era5_xy
    mock_make_tiles.return_value = mock_tiles
    mock_gen_patches.return_value = (mock_patches, mock_metadata)

    from super_resolution.pre_processing_module.preprocessing import run_preprocessing

    data_paths = {"era5": "/era5.nc", "rcm": "/rcm.zarr"}
    with suppress_type_checks():
        input_patches, metadata, tiles, rcm_ds, gcm_ds = run_preprocessing(data_paths)

    mock_load_era5.assert_called_once_with("/era5.nc")
    mock_load_rcm.assert_called_once_with("/rcm.zarr")
    mock_create_grid.assert_called_once_with(mock_rcm, resx_km=14, resy_km=28)
    mock_reproject.assert_called_once_with(mock_gcm, mock_gridx, mock_gridy)
    mock_make_tiles.assert_called_once()
    mock_gen_patches.assert_called_once_with(mock_era5_xy, mock_tiles)
    assert input_patches is mock_patches
    assert metadata is mock_metadata
    assert tiles is mock_tiles
    assert rcm_ds is mock_rcm
    assert gcm_ds is mock_era5_xy
