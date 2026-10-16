"""Tests for the attentionUNet module."""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Component-level tests (RCM_Emulator_parts)
# ---------------------------------------------------------------------------


def test_double_conv_ds_output_shape():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_parts import DoubleConvDS

    conv = DoubleConvDS(in_channels=32, out_channels=64, kernels_per_layer=2)
    x = torch.zeros(1, 32, 16, 16)
    out = conv(x)
    assert out.shape == (1, 64, 16, 16)


def test_down_ds_output_shape():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_parts import DownDS

    down = DownDS(in_channels=32, out_channels=64)
    x = torch.zeros(1, 32, 16, 16)
    out = down(x)
    assert out.shape == (1, 64, 8, 8)


def test_up_last_conv_attn_output_shape():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_parts import UpLastConv

    up = UpLastConv(in_channels=32, out_channels=16, mid_channels=16, bilinear=False)
    x = torch.zeros(1, 32, 32, 32)
    out = up(x, zsize=64)
    assert out.shape[0] == 1
    assert out.shape[1] == 16


def test_linear_upsampling_z_attn():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_parts import LinearUpsamplingZ

    linz = LinearUpsamplingZ(in_channels=16, out_channels=1024)
    z = torch.zeros(2, 16, 1, 1)
    out = linz(z)
    assert out.shape == (2, 1024, 1, 1)


def test_cbam_output_shape():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_parts import CBAM

    cbam = CBAM(input_channels=32, reduction_ratio=8)
    x = torch.rand(1, 32, 16, 16)
    out = cbam(x)
    assert out.shape == (1, 32, 16, 16)


# ---------------------------------------------------------------------------
# SmaAt_UNet instantiation
# ---------------------------------------------------------------------------


def test_smaat_unet_instantiation():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_unet import SmaatUnet

    model = SmaatUnet(n_channels_x=7, n_channels_z=16, bilinear=False)
    assert isinstance(model, nn.Module)
    assert model.n_channels_x == 7
    assert model.n_channels_z == 16


def test_smaat_unet_has_expected_layers():
    from super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_unet import SmaatUnet

    model = SmaatUnet(n_channels_x=7, n_channels_z=16)
    assert hasattr(model, "inc")
    assert hasattr(model, "linz")
    assert hasattr(model, "down1")
    assert hasattr(model, "down2")
    assert hasattr(model, "cbam1")
    assert hasattr(model, "final")
