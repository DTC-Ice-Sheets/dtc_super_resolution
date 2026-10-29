"""Tests for the SmaatUnet model in the attention_unet module."""

import torch


def test_smaat_unet_forward_output_shape():
    from dtc_super_resolution.rcm_emulator.ml_module.attention_unet.rcm_emulator_unet import SmaatUnet

    model = SmaatUnet(n_channels_x=7, n_channels_z=16)
    model.eval()
    x = torch.randn(1, 7, 64, 64)
    z = torch.randn(1, 16, 1, 1)

    with torch.no_grad():
        output = model(x, z)

    assert output.shape == (1, 224, 224)
    assert torch.isfinite(output).all()
