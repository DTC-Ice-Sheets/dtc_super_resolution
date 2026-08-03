"""Tests for the ml_module datasets module."""

import numpy as np
import torch

from super_resolution.ml_module.datasets import (
    RACMO_TRAIN_MEAN,
    RACMO_TRAIN_STD,
    InferenceDataset,
    inverse_signed_log1p,
    signed_log1p,
    standardize,
    unstandardize,
)

# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------


def test_standardize_zero_mean_unit_std():
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    result = standardize(data, ownstd=True)
    assert result.shape == data.shape
    # After standardization mean should be ~0
    assert abs(result.mean()) < 1e-5


def test_standardize_handles_nan():
    data = np.array([[[np.nan, 2.0], [3.0, 4.0]]])
    result = standardize(data, ownstd=True)
    assert not np.any(np.isnan(result))


def test_standardize_with_provided_mean_std():
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    result = standardize(data, mean=np.array([2.5]), std=np.array([1.0]), ownstd=False)
    expected = (data - 2.5) / (1.0 + 1e-8)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_standardize_constant_array_no_nan():
    """Constant channel: std = 0 → divide by eps, result should be 0 (all same, no NaN)."""
    data = np.ones((1, 4, 4))
    result = standardize(data, ownstd=True)
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# unstandardize
# ---------------------------------------------------------------------------


def test_unstandardize_inverse_of_precomputed_std():
    data = np.array([0.0, 1.0, -1.0])
    result = unstandardize(data)
    expected = data * RACMO_TRAIN_STD + RACMO_TRAIN_MEAN
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# signed_log1p / inverse_signed_log1p
# ---------------------------------------------------------------------------


def test_signed_log1p_positive():
    x = torch.tensor([1.0, 2.0, 3.0])
    result = signed_log1p(x)
    assert (result > 0).all()


def test_signed_log1p_negative():
    x = torch.tensor([-1.0, -2.0])
    result = signed_log1p(x)
    assert (result < 0).all()


def test_inverse_signed_log1p_roundtrip():
    x = torch.tensor([0.5, -0.5, 2.0, -3.0])
    transformed = signed_log1p(x)
    recovered = inverse_signed_log1p(transformed)
    torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# InferenceDataset
# ---------------------------------------------------------------------------


def _make_patches(n=3, c=8, h=6, w=8):
    """Create synthetic ERA5 patches. Channel 3 (SST) is 0 to simulate land."""
    patches = np.random.rand(n, c, h, w).astype(np.float32)
    patches[:, 3, :, :] = 0.0  # SST = 0 (land)
    return patches


def _make_metadata(n=3):
    return [{"time_idx": np.datetime64(f"2021-0{i + 1}-01", "D"), "tile_idx": i} for i in range(n)]


def test_inference_dataset_length():
    patches = _make_patches(n=5)
    metadata = _make_metadata(n=5)
    ds = InferenceDataset(patches, metadata, ownstd=True)
    assert len(ds) == 5


def test_inference_dataset_getitem_shapes():
    patches = _make_patches(n=2, c=8, h=6, w=8)
    metadata = _make_metadata(n=2)
    ds = InferenceDataset(patches, metadata, ownstd=True)

    x, z = ds[0]

    assert x.shape == (8, 6, 8)
    assert z.ndim == 3
    assert z.shape[1] == 1
    assert z.shape[2] == 1


def test_inference_dataset_no_nan_in_output():
    patches = _make_patches(n=3, c=8, h=6, w=8)
    metadata = _make_metadata(n=3)
    ds = InferenceDataset(patches, metadata, ownstd=True)

    for i in range(len(ds)):
        x, z = ds[i]
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(z))


def test_inference_dataset_month_encoding_range():
    """cos/sin month encoding values should be in [-1, 1]."""
    patches = _make_patches(n=12, c=8, h=4, w=4)
    metadata = [{"time_idx": np.datetime64(f"2021-{str(m + 1).zfill(2)}-01", "D"), "tile_idx": m} for m in range(12)]
    ds = InferenceDataset(patches, metadata, ownstd=True)

    for i in range(len(ds)):
        _, z = ds[i]
        # last two 'channels' in Z are cos and sin month values
        assert z.min() >= -1.1
        assert z.max() <= 1.1


# ---------------------------------------------------------------------------
# createMask
# ---------------------------------------------------------------------------


def test_create_mask_all_zeros_is_sea():
    from super_resolution.ml_module.datasets import create_mask

    smb = np.zeros((4, 5))  # all zeros → sea
    mask = create_mask(smb)
    assert mask.shape == (4, 5)
    assert (mask == 0).all()


def test_create_mask_nonzero_is_land():
    from super_resolution.ml_module.datasets import create_mask

    smb = np.ones((3, 4))  # all non-zero → land
    mask = create_mask(smb)
    assert (mask == 1).all()


def test_create_mask_mixed():
    from super_resolution.ml_module.datasets import create_mask

    smb = np.array([[1.0, 0.0], [0.0, 2.0]])
    mask = create_mask(smb)
    assert mask[0, 0] == 1  # non-zero → land
    assert mask[1, 0] == 0  # zero → sea
    assert mask[0, 1] == 0  # zero → sea
    assert mask[1, 1] == 1  # non-zero → land
