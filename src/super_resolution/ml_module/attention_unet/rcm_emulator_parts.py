#!/usr/bin/env python3

"""Parts of the U-Net model."""

# Base model taken from: https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as functional


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpLastConv(nn.Module):
    """Upscaling then double conv without concatenating with skip connection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        bilinear=True,
        kernels_per_layer=1,
    ) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
            # self.conv = DoubleConv(mid_channels, out_channels)
            self.conv = DoubleConvDS(mid_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x, zsize=64):
        x1 = self.up(x)
        # input is CHW
        diff_y = zsize - x1.size()[2]
        diff_x = zsize - x1.size()[3]

        x1 = functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(x1)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InitialConv(nn.Module):
    """Initial convolution to adapt spatial dimensions."""

    def __init__(self, in_channels, out_channels, size_in, size_out) -> None:
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

    def forward(self, x):
        return self.adapt(x)


# class InitialConv(nn.Module):
#     def __init__(self, in_channels, out_channels, sizex, size):
#         super().__init__()
#         self.diff = sizex - size + 1
#         self.n_channels_in = in_channels
#         self.n_channels_out = out_channels
#         self.firstconv = nn.Sequential(
#             nn.Conv2d(self.n_channels_in, self.n_channels_out, kernel_size=self.diff),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x1 = self.firstconv(x)
#         return x1


# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
class DepthToSpace(nn.Module):
    def __init__(self, block_size) -> None:
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs**2), h, w)  # (n, bs, bs, c//bs^2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (n, c//bs^2, h, bs, w, bs)
        return x.view(n, c // (self.bs**2), h * self.bs, w * self.bs)  # (n, c//bs^2, h * bs, w * bs)


class SpaceToDepth(nn.Module):
    # Expects the following shape: Batch, Channel, Height, Width
    def __init__(self, block_size) -> None:
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)  # (n, c, h//bs, bs, w//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (n, bs, bs, c, h//bs, w//bs)
        return x.view(n, c * (self.bs**2), h // self.bs, w // self.bs)  # (n, c*bs^2, h//bs, w//bs)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1) -> None:
        super().__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels) -> None:
        super().__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    def forward(self, x):
        out = functional.relu(self.dense1(x.view(x.size(0), -1)))
        out = functional.relu(self.dense2(out))
        return self.dense3(out)


class DoubleDSConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_ds_conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        return x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.Identity()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        return self.spatial_att(out)


class LinearUpsamplingZ(nn.Module):
    # Linear upsampling of z to same number of channels as downsampled X
    # so that can concatenate
    def __init__(self, in_channels=16, out_channels=1024) -> None:
        super().__init__()
        self.linearUp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
        )

    def forward(self, z):
        # transform to remove last dimensions
        z1 = z.view(z.shape[0], -1)  # (batch, 16)
        z5 = self.linearUp(z1.float())  # (batch, 1024)
        # transform back to original shape
        return z5.view(z5.shape[0], 1, 1, -1).transpose(3, 1)  # (batch, 1024, 1, 1)
