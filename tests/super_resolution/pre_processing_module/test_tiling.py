"""Tests for the tiling module."""

from unittest.mock import MagicMock

from dtc_super_resolution.rcm_emulator.pre_processing_module.tiling import generate_tile_indices


def make_fake_grid(h, w):
    """Create a mock xr.DataArray-like object exposing only .rio.shape."""
    fake = MagicMock()
    fake.rio.shape = (h, w)
    return fake


class TestGenerateTileIndicesBasic:
    def test_non_overlapping_tiles_exact_fit(self):
        racmo = make_fake_grid(h=20, w=20)
        era5 = make_fake_grid(h=10, w=10)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=2,
            scale_y=2,
            stride_frac=1.0,
            context_pad=(0, 0),
        )

        assert len(tiles) == 4
        assert tiles[0]["racmo_slice_y"] == slice(0, 10)
        assert tiles[0]["racmo_slice_x"] == slice(0, 10)
        assert tiles[0]["era5_slice_y"] == slice(0, 5)
        assert tiles[0]["era5_slice_x"] == slice(0, 5)

    def test_era5_mapping_uses_correct_scale(self):
        racmo = make_fake_grid(h=10, w=10)
        era5 = make_fake_grid(h=5, w=5)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=2,
            scale_y=2,
            stride_frac=1.0,
            context_pad=(0, 0),
        )

        assert len(tiles) == 1
        assert tiles[0]["era5_slice_y"] == slice(0, 5)
        assert tiles[0]["era5_slice_x"] == slice(0, 5)

    def test_no_tiles_when_grid_smaller_than_tile(self):
        racmo = make_fake_grid(h=5, w=5)
        era5 = make_fake_grid(h=3, w=3)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=2,
            scale_y=2,
            stride_frac=1.0,
            context_pad=(0, 0),
        )

        assert tiles == []


class TestStrideSnapping:
    def test_stride_frac_half_produces_overlap(self):
        racmo = make_fake_grid(h=20, w=20)
        era5 = make_fake_grid(h=10, w=10)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=2,
            scale_y=2,
            stride_frac=0.5,
            context_pad=(0, 0),
        )

        ys = sorted({t["racmo_slice_y"].start for t in tiles})
        assert ys[1] - ys[0] == 4  # stride snapped down to nearest multiple of scale_y (2)

    def test_stride_never_below_scale(self):
        racmo = make_fake_grid(h=20, w=20)
        era5 = make_fake_grid(h=10, w=10)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=2,
            scale_y=2,
            stride_frac=0.01,
            context_pad=(0, 0),
        )

        ys = sorted({t["racmo_slice_y"].start for t in tiles})
        assert ys[1] - ys[0] == 2  # clamped to scale_y minimum


class TestContextPadding:
    def test_context_pad_expands_era5_patch(self):
        # 40x40 grids: boundary tiles at i=0 and i=30 are dropped by pad=2;
        # first surviving interior tile is at RACMO i=10, j=10 → ERA5 ei=10, ej=10.
        racmo = make_fake_grid(h=40, w=40)
        era5 = make_fake_grid(h=40, w=40)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=1,
            scale_y=1,
            stride_frac=1.0,
            context_pad=(2, 2),
        )

        assert tiles[0]["era5_slice_y"] == slice(10 - 2, 10 + 10 + 2)
        assert tiles[0]["era5_slice_x"] == slice(10 - 2, 10 + 10 + 2)

    def test_out_of_bounds_padded_tiles_are_dropped(self):
        # ERA5 grid is only just big enough for the tile with no room for padding,
        # so every candidate tile's padded patch should exceed bounds and be dropped.
        racmo = make_fake_grid(h=10, w=10)
        era5 = make_fake_grid(h=10, w=10)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=1,
            scale_y=1,
            stride_frac=1.0,
            context_pad=(2, 2),
        )

        assert tiles == []

    def test_context_pad_zero_matches_unpadded_case(self):
        racmo = make_fake_grid(h=20, w=20)
        era5 = make_fake_grid(h=20, w=20)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=1,
            scale_y=1,
            stride_frac=1.0,
            context_pad=(0, 0),
        )

        assert tiles[0]["era5_slice_y"] == slice(0, 10)
        assert tiles[0]["era5_slice_x"] == slice(0, 10)

    def test_asymmetric_context_pad_h_and_w(self):
        # To catch bugs where H and W padding get swapped.
        # 30x30 grids: boundary tiles are dropped; the only surviving interior
        # tile is at RACMO i=10, j=10 → ERA5 ei=10, ej=10.
        # pad_h=1 expands y by 1 on each side; pad_w=3 expands x by 3 on each side.
        racmo = make_fake_grid(h=30, w=30)
        era5 = make_fake_grid(h=30, w=30)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=1,
            scale_y=1,
            stride_frac=1.0,
            context_pad=(1, 3),
        )

        assert tiles[0]["era5_slice_y"] == slice(10 - 1, 10 + 10 + 1)
        assert tiles[0]["era5_slice_x"] == slice(10 - 3, 10 + 10 + 3)

    def test_context_pad_drops_only_boundary_tiles(self):
        # Interior tiles should survive padding; only edge tiles touching
        # the array boundary should be dropped.
        racmo = make_fake_grid(h=40, w=40)
        era5 = make_fake_grid(h=40, w=40)

        tiles = generate_tile_indices(
            racmo,
            era5,
            tile_racmo_h=10,
            tile_racmo_w=10,
            scale_x=1,
            scale_y=1,
            stride_frac=1.0,
            context_pad=(2, 2),
        )

        # 4x4 grid of candidate tiles; edge row/col (i=0 or i=30, j=0 or j=30)
        # should be dropped due to padding, leaving only the interior 2x2 = 4 tiles.
        assert len(tiles) == 4
        for t in tiles:
            assert t["era5_slice_y"].start >= 0
            assert t["era5_slice_x"].start >= 0
