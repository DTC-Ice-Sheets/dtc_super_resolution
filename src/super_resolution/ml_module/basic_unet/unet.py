"""Full assembly of the parts to form the complete network."""

# Source UNet: https://github.com/milesial/Pytorch-UNet
import torch
from torch import nn

from .unet_parts import DoubleConv, Down, InitialConv, LinearUpsamplingZ, Up, UpLastConv


class UNetBaseline(nn.Module):
    def __init__(self, n_channels_x: int = 8, n_channels_z: int = 18, size: int = 64, filter: int = 64, bilinear: bool = False) -> None:
        """Initialize the UNetBaseline."""
        super().__init__()
        self.n_channels_x = n_channels_x
        self.n_channels_z = n_channels_z
        self.size = size
        self.filter = filter
        self.bilinear = bilinear

        # input  (batch, 8, 48, 48)
        # first conv:
        self.initialconv = InitialConv(self.n_channels_x, 32, 48, self.size)

        # downsampling:
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)
        self.d5 = Down(512, 1024)

        # last conv of downsampling
        self.last_conv_down = DoubleConv(1024, 1024)

        # linewar downsampling of z:
        self.linz = LinearUpsamplingZ(self.n_channels_z, 1024)

        # upsampling:
        self.up1 = Up(2048, 1024, self.bilinear)
        self.up2 = Up(1024, 512, self.bilinear)
        self.up3 = Up(512, 256, self.bilinear)
        self.up4 = Up(256, 128, self.bilinear)
        self.up5 = Up(128, 64, self.bilinear)
        # post-decoder upsampling: 64x64 -> 128x128 -> 256x256
        self.up6 = UpLastConv(64, 32, 32, self.bilinear)
        self.up7 = UpLastConv(32, 16, 16, self.bilinear)

        # linear spatial adapter: 256x256 -> 224x224, 16ch -> 1ch
        self.final = nn.Conv2d(16, 1, kernel_size=33)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for the UNetBaseline."""
        # input  (batch, 7, 32, 32)
        x1 = self.initialconv(x)  # (batch, 32, 64, 64)

        # Downsampling
        x2, x2_bm = self.d1(x1)  # (batch, 64, 32, 32)
        x3, x3_bm = self.d2(x2)  # (batch, 128, 16, 16)
        x4, x4_bm = self.d3(x3)  # (batch, 256, 8, 8)
        x5, x5_bm = self.d4(x4)  # (batch, 512, 4, 4)
        # x6, x6_bm = self.d5(x5)  # (batch, 1024, 2, 2)
        x7 = self.last_conv_down(x5)  # (batch, 1024, 2, 2)

        # Upsample second input to have same shape as last_conv [1024, 1, 1]
        z1 = self.linz(z)

        # expand z1 spatially to match x7 (handles any bottleneck size, e.g. 2x2 when size=64)
        z1 = z1.expand(-1, -1, x7.size(2), x7.size(3))

        # concatenate both:
        x8 = torch.cat([x7, z1], dim=1)  # (batch, 2048, H, H)

        # Upsampling (decoder)
        x = self.up1(x8, x5_bm)  # (batch, 1024, 4, 4)
        # x = self.up2(x, x5_bm)   # (batch, 512,  8, 8)
        x = self.up3(x, x4_bm)  # (batch, 256, 16, 16)
        x = self.up4(x, x3_bm)  # (batch, 128, 32, 32)
        x = self.up5(x, x2_bm)  # (batch,  64, 64, 64)

        # Post-decoder upsampling
        x = self.up6(x, zsize=128)  # (batch, 32, 128, 128)
        x = self.up7(x, zsize=256)  # (batch, 16, 256, 256)

        # Linear spatial adapter: 256x256 -> 224x224, 16ch -> 1ch
        x = self.final(x)  # (batch,  1, 224, 224)
        return x.squeeze(1)  # (batch, 224, 224)
