"""Tests for the basicUNet module."""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Component-level tests (unet_parts)
# ---------------------------------------------------------------------------


def test_double_conv_output_shape():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import DoubleConv

    conv = DoubleConv(in_channels=4, out_channels=8)
    x = torch.zeros(1, 4, 16, 16)
    out = conv(x)
    assert out.shape == (1, 8, 16, 16)


def test_down_output_shapes():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import Down

    down = Down(in_channels=8, out_channels=16)
    x = torch.zeros(1, 8, 16, 16)
    downsampled, skip = down(x)
    # downsampled is halved spatially
    assert downsampled.shape == (1, 16, 8, 8)
    # skip connection retains original spatial size
    assert skip.shape == (1, 16, 16, 16)


def test_up_output_shape():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import Up

    # With bilinear=False, Up uses ConvTranspose2d(in_channels, in_channels//2).
    # x1 must have in_channels channels; x2 has in_channels//2 channels.
    in_ch = 32
    up = Up(in_channels=in_ch, out_channels=16, bilinear=False)
    x1 = torch.zeros(1, in_ch, 4, 4)  # input to be upsampled
    x2 = torch.zeros(1, in_ch // 2, 8, 8)  # skip connection
    out = up(x1, x2)
    assert out.shape[0] == 1
    assert out.shape[1] == 16


def test_initial_conv_downsampling():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import InitialConv

    conv = InitialConv(in_channels=7, out_channels=32, size_in=48, size_out=64)
    x = torch.zeros(1, 7, 48, 48)
    out = conv(x)
    assert out.shape == (1, 32, 64, 64)


def test_initial_conv_upsampling():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import InitialConv

    # size_in < size_out → upsampling path
    conv = InitialConv(in_channels=7, out_channels=32, size_in=32, size_out=64)
    x = torch.zeros(1, 7, 32, 32)
    out = conv(x)
    assert out.shape == (1, 32, 64, 64)


def test_linear_upsampling_z():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import LinearUpsamplingZ

    linz = LinearUpsamplingZ(in_channels=16, out_channels=1024)
    z = torch.zeros(2, 16, 1, 1)
    out = linz(z)
    assert out.shape == (2, 1024, 1, 1)


def test_out_conv_output_shape():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import OutConv

    conv = OutConv(in_channels=16, out_channels=1)
    x = torch.zeros(1, 16, 32, 32)
    out = conv(x)
    assert out.shape == (1, 1, 32, 32)


def test_up_last_conv_output_shape():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet_parts import UpLastConv

    up = UpLastConv(in_channels=64, out_channels=32, mid_channels=32, bilinear=False)
    x = torch.zeros(1, 64, 32, 32)
    out = up(x, zsize=64)
    assert out.shape[0] == 1
    assert out.shape[1] == 32


# ---------------------------------------------------------------------------
# UNetBaseline instantiation
# ---------------------------------------------------------------------------


def test_unet_baseline_instantiation():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet import UNetBaseline

    model = UNetBaseline(n_channels_x=7, n_channels_z=16, size=64, n_filter=64, bilinear=False)
    assert isinstance(model, nn.Module)
    assert model.n_channels_x == 7
    assert model.n_channels_z == 16


def test_unet_baseline_has_expected_layers():
    from dtc_super_resolution.rcm_emulator.ml_module.basic_unet.unet import UNetBaseline

    model = UNetBaseline(n_channels_x=7, n_channels_z=16, size=64, n_filter=64, bilinear=False)
    assert hasattr(model, "initialconv")
    assert hasattr(model, "d1")
    assert hasattr(model, "d2")
    assert hasattr(model, "d3")
    assert hasattr(model, "d4")
    assert hasattr(model, "linz")
    assert hasattr(model, "final")
