"""Parts of the U-Net model."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 0) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the DoubleConv module."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with double conv then maxpool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
        self.maxpool = nn.MaxPool2d(2)
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Down module."""
        x1 = self.doubleconv(x)
        x2 = self.maxpool(x1)
        return x2, x1


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Up module."""
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpLastConv(nn.Module):
    """Upscaling then double conv without concatenating with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
            self.conv = DoubleConv(mid_channels, out_channels)

    def forward(self, x: torch.Tensor, zsize: int = 64) -> torch.Tensor:
        """Forward pass for the UpLastConv module."""
        x1 = self.up(x)
        # input is CHW
        diff_y = zsize - x1.size()[2]
        diff_x = zsize - x1.size()[3]

        x1 = functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(x1)


class InitialConv(nn.Module):
    """Initial convolution to adapt spatial dimensions."""

    def __init__(self, in_channels: int, out_channels: int, size_in: int, size_out: int) -> None:
        super().__init__()
        self.n_channels_in = in_channels
        self.n_channels_out = out_channels
        if size_in >= size_out:
            # downsampling via valid convolution
            diff = size_in - size_out + 1
            self.adapt = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=diff),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # upsampling via bilinear resize + 3x3 conv
            self.adapt = nn.Sequential(
                nn.Upsample(size=size_out, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the InitialConv module."""
        return self.adapt(x)


# class InitialConv(nn.Module):
#       """
#       Initial convolution to reduce spatial dimensions
#       """
#       def __init__(self, in_channels: int, out_channels: int, sizex: int, size: int) -> None:
#               super().__init__()
#               self.diff = sizex-size+1
#               self.n_channels_in = in_channels
#               self.n_channels_out = out_channels
#               self.firstconv = nn.Sequential(
#                       nn.Conv2d(self.n_channels_in, self.n_channels_out, kernel_size=self.diff),
#                       nn.BatchNorm2d(out_channels),
#                       nn.ReLU(inplace=True)
#               )
#       def forward(self, x):
#               x1 = self.firstconv(x)
#               return x1


class OutConv(nn.Module):
    """Final convolution to reduce channels to 1."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the OutConv module."""
        return self.conv(x)


class LinearUpsamplingZ(nn.Module):
    """Linear upsampling of z to match the number of channels of downsampled x."""

    # Linear upsampling of z to same number of channels as downsampled X
    # so that can concatenate
    def __init__(self, in_channels: int = 16, out_channels: int = 1024) -> None:
        super().__init__()

        if out_channels == 0:
            out_channels = 1024  # default to 64x upsampling

        self.linearUp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, out_channels)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for linear upsampling of z."""
        # transform to remove last dimensions
        z1 = z.view(z.shape[0], -1)  # (batch, 16)
        z5 = self.linearUp(z1.float())  # (batch, 1024)
        # transform back to original shape
        return z5.view(z5.shape[0], 1, 1, -1).transpose(3, 1)  # (batch, 1024, 1, 1)
