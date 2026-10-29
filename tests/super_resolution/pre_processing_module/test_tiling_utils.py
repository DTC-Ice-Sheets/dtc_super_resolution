"""Tests for tiling utility functions."""

import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from typeguard import suppress_type_checks


def _make_grid_dataset(nx, ny, res_m=2000.0):
    x = np.arange(nx, dtype=float) * res_m
    y = np.arange(ny, dtype=float) * res_m
    data = np.ones((ny, nx), dtype=np.float32)
    ds = xr.Dataset(
        {"surface_mass_balance_glaciated": (["y", "x"], data)},
        coords={"x": x, "y": y},
    )
    return ds.rio.write_crs("epsg:3031")


def _make_era5_grid(nx, ny, res_x=14000.0, res_y=28000.0):
    x = np.arange(nx, dtype=float) * res_x
    y = np.arange(ny, dtype=float) * res_y
    data = np.ones((ny, nx), dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (["y", "x"], data)},
        coords={"x": x, "y": y},
    )
    return ds.rio.write_crs("epsg:3031")


# ---------------------------------------------------------------------------
# make_tiles with log_=True
# ---------------------------------------------------------------------------


@patch("dtc_super_resolution.rcm_emulator.pre_processing_module.tiling.generate_tile_indices")
def test_make_tiles_log_branch(mock_gen, caplog):
    import logging

    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import make_tiles

    mock_gen.return_value = [{"tile": 1}]

    racmo = _make_grid_dataset(nx=60, ny=60, res_m=2000.0)
    era5 = _make_era5_grid(nx=10, ny=10)

    with suppress_type_checks(), caplog.at_level(logging.INFO):
        result = make_tiles(era5, racmo, tile_size=4, context_pad_h=1, context_pad_w=1, stride_frac=1.0, log_=True)

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# slice_to_dict / dict_to_slice
# ---------------------------------------------------------------------------


def test_slice_to_dict():
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import slice_to_dict

    s = slice(2, 10, None)
    result = slice_to_dict(s)
    assert result == {"start": 2, "stop": 10, "step": None}


def test_dict_to_slice():
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import dict_to_slice

    d = {"start": 3, "stop": 15, "step": None}
    result = dict_to_slice(d)
    assert result == slice(3, 15, None)


def test_slice_roundtrip():
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import dict_to_slice, slice_to_dict

    original = slice(5, 20, None)
    assert dict_to_slice(slice_to_dict(original)) == original


# ---------------------------------------------------------------------------
# save_tiles_metadata / load_tiles_metadata
# ---------------------------------------------------------------------------


def _sample_tiles():
    return [
        {
            "racmo_slice_y": slice(0, 10),
            "racmo_slice_x": slice(0, 15),
            "era5_slice_y": slice(0, 5),
            "era5_slice_x": slice(0, 8),
        },
        {
            "racmo_slice_y": slice(10, 20),
            "racmo_slice_x": slice(15, 30),
            "era5_slice_y": slice(5, 10),
            "era5_slice_x": slice(8, 16),
        },
    ]


def test_save_tiles_metadata_creates_json(tmp_path):
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import save_tiles_metadata

    tiles = _sample_tiles()
    fpath = str(tmp_path / "tiles.json")
    save_tiles_metadata(tiles, fpath)

    assert pathlib.Path(fpath).exists()
    with pathlib.Path(fpath).open() as f:
        data = json.load(f)
    assert len(data) == 2


def test_load_tiles_metadata_roundtrip(tmp_path):
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import (
        load_tiles_metadata,
        save_tiles_metadata,
    )

    tiles = _sample_tiles()
    fpath = str(tmp_path / "tiles.json")
    save_tiles_metadata(tiles, fpath)

    loaded = load_tiles_metadata(fpath)

    assert len(loaded) == 2
    assert loaded[0]["racmo_slice_y"] == tiles[0]["racmo_slice_y"]
    assert loaded[1]["era5_slice_x"] == tiles[1]["era5_slice_x"]


# ---------------------------------------------------------------------------
# clip_data_to_intersection
# ---------------------------------------------------------------------------


def test_clip_data_to_intersection():
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import clip_data_to_intersection

    racmo = _make_grid_dataset(nx=20, ny=20, res_m=2000.0)
    era5 = _make_era5_grid(nx=20, ny=15, res_x=2000.0, res_y=2000.0)

    with suppress_type_checks():
        clipped_racmo, clipped_era5 = clip_data_to_intersection(racmo, era5)

    assert clipped_racmo is not None
    assert clipped_era5 is not None


# ---------------------------------------------------------------------------
# save_tiles_to_disk
# ---------------------------------------------------------------------------


@patch("dtc_super_resolution.rcm_emulator.pre_processing_module.tiling.xr.open_zarr")
def test_save_tiles_to_disk_creates_files(mock_open_zarr, tmp_path):
    from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import save_tiles_to_disk

    ny, nx = 40, 40
    h_era5, w_era5 = 10, 10

    import pandas as pd

    times = pd.DatetimeIndex(["2021-01-01"])

    # Build a minimal RACMO zarr mock
    racmo_data = np.ones((1, ny, nx), dtype=np.float32)
    year_ds = xr.Dataset(
        {"surface_mass_balance_glaciated": (["time", "y", "x"], racmo_data)},
        coords={
            "x": np.arange(nx, dtype=float) * 2000.0,
            "y": np.arange(ny, dtype=float) * 2000.0,
            "time": times,
        },
    )
    year_ds = year_ds.rio.write_crs("epsg:3031")
    mock_open_zarr.return_value = MagicMock()
    mock_open_zarr.return_value.rio.write_crs.return_value = year_ds

    # ERA5 dataset with matching timestamps
    era5_arr = np.random.rand(1, h_era5, w_era5).astype(np.float32)
    era5 = xr.Dataset(
        {
            "t2m": (["time", "y", "x"], era5_arr),
            "sst": (["time", "y", "x"], era5_arr),
        },
        coords={
            "time": times,
            "x": np.arange(w_era5, dtype=float) * 14000.0,
            "y": np.arange(h_era5, dtype=float) * 14000.0,
        },
    )

    tiles = [
        {
            "racmo_slice_y": slice(0, 20),
            "racmo_slice_x": slice(0, 20),
            "era5_slice_y": slice(0, 5),
            "era5_slice_x": slice(0, 5),
        }
    ]

    out_dir = str(tmp_path / "tiles")
    with suppress_type_checks():
        save_tiles_to_disk(
            era5=era5,
            racmo_files=["/fake/racmo.zarr"],
            tiles=tiles,
            output_dir=out_dir,
            max_nan_frac=0.95,
        )

    assert pathlib.Path(out_dir).is_dir()
