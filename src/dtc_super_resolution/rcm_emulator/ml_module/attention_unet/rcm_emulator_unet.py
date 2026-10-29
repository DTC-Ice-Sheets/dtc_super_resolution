"""SmaAt-UNet model for super-resolution of climate data with attention mechanisms."""

import torch
from torch import nn

from .rcm_emulator_parts import CBAM, DownDS, InitialConv, LinearUpsamplingZ, UpDS, UpLastConv


class SmaatUnet(nn.Module):
    """SmaAt-UNet model for super-resolution of climate data with attention mechanisms."""

    def __init__(
        self,
        n_channels_x: int,
        n_channels_z: int,
        kernels_per_layer: int = 2,
        bilinear: bool = False,
        reduction_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.n_channels_x = n_channels_x
        self.n_channels_z = n_channels_z
        self.size = 64
        # self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        # input  (batch, 7, 32, 32)
        # first conv:
        self.inc = InitialConv(self.n_channels_x, 32, 48, self.size)

        # linewar downsampling of z:
        self.linz = LinearUpsamplingZ(self.n_channels_z, 1024)

        # self.inc = DoubleConvDS(self.n_channels_x, 32, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(32, 64, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down4 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(512, reduction_ratio=reduction_ratio)

        factor = 2 if self.bilinear else 1
        self.down5 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam6 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.cbam6z = CBAM(1024, reduction_ratio=reduction_ratio)  # separate CBAM for z input

        self.up1 = UpDS(2048, 1024 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up5 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up6 = UpDS(64, 32, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.up7 = UpLastConv(32, 32, 16, self.bilinear)
        self.up8 = UpLastConv(32, 16, 16, self.bilinear)
        self.up9 = UpLastConv(16, 16, 8, self.bilinear)

        # final layer: 256x256 -> 224x224, 16ch -> 1ch
        self.final = nn.Conv2d(16, 1, kernel_size=33)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SmaAt_UNet."""
        # input  (batch, 7, 32, 32)
        x1 = self.inc(x)  # (batch, 32, 32, 32)
        x1_att = self.cbam1(x1)  # (batch, 32, 32, 32)

        # downsampling
        x2 = self.down1(x1)  # (batch, 64, 16, 16)
        x2_att = self.cbam2(x2)
        x3 = self.down2(x2)  # (batch, 128, 8, 8)
        x3_att = self.cbam3(x3)
        x4 = self.down3(x3)  # (batch, 256, 4, 4)
        x4_att = self.cbam4(x4)
        x5 = self.down4(x4)  # (batch, 512, 2, 2)
        x5_att = self.cbam5(x5)
        x6 = self.down5(x5)  # (batch, 1024, 1, 1)
        x6_att = self.cbam6(x6)

        # Expand z to match bottleneck spatial size (e.g. 2x2 when size=64)
        z1 = self.linz(z)
        z1 = z1.expand(-1, -1, x6.size(2), x6.size(3))
        # attention:
        z1_att = self.cbam6z(z1)

        # concatenate both:
        x8 = torch.cat([x6_att, z1_att], dim=1)

        x = self.up1(x8, x6_att)
        x = self.up2(x, x5_att)  # (batch, 512, 2, 2)
        x = self.up3(x, x4_att)  # (batch, 256, 4, 4)
        x = self.up4(x, x3_att)  # (batch, 128, 8, 8)
        x = self.up5(x, x2_att)  # (batch, 64, 16, 16)
        x = self.up6(x, x1_att)  # (batch, 32, 32, 32)

        x = self.up7(x)  # (batch, 32, 64, 64)
        x = self.up8(x, zsize=128)  # (batch, 16, 128, 128)
        x = self.up9(x, zsize=256)  # (batch, 16, 256, 256)
        x = self.final(x)  # (batch, 1, 224, 224)
        return x.squeeze(1)  # (batch, 224, 224)
