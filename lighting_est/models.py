import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import srgb_to_linear


def densenet_weights(value):
    return (
        None if value in (None, "none", "None") else models.DenseNet161_Weights.DEFAULT
    )


class IDResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        groups = min(32, out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class IDDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.block = IDResidualBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class IDUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = IDResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class IDContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList(
            [
                IDResidualBlock(channels, branch_channels, dilation=dilation)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.fuse = IDResidualBlock(branch_channels * 4, channels)

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class IDNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.encoder1 = IDResidualBlock(n_channels, 64)
        self.encoder2 = IDDown(64, 128)
        self.encoder3 = IDDown(128, 256)
        self.encoder4 = IDDown(256, 512)
        self.bottleneck = nn.Sequential(
            IDDown(512, 768),
            IDContext(768),
        )
        self.decoder4 = IDUp(768, 512, 512)
        self.decoder3 = IDUp(512, 256, 256)
        self.decoder2 = IDUp(256, 128, 128)
        self.decoder1 = IDUp(128, 64, 64)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def padded(self, x):
        height, width = x.shape[-2:]
        pad_height = (-height) % 16
        pad_width = (-width) % 16
        if not pad_height and not pad_width:
            return x
        can_reflect = pad_height < height and pad_width < width
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (0, pad_width, 0, pad_height), mode=mode)

    def forward(self, x):
        height, width = x.shape[-2:]
        x = self.padded(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = self.bottleneck(x4)
        x = self.decoder4(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        return torch.sigmoid(self.output(x)[..., :height, :width])


class SGNet(nn.Module):
    def __init__(self, sg_num=12, backbone_weights="DEFAULT"):
        super().__init__()
        self.sg_num = sg_num
        self.min_positive = 1e-4

        densenet = models.densenet161(weights=densenet_weights(backbone_weights))
        self.encoder = densenet.features

        self.mask_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(2208 + 128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        in_features = 128 * 1 * 1
        self.head_p = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 3))
        self.head_la = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 1))
        self.head_w = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num))

    def forward(self, x, lum):
        lum_features = self.mask_processor(lum)

        x = self.encoder(x)
        lum_features = nn.functional.interpolate(
            lum_features,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, lum_features], dim=1)
        x = self.decoder_conv(x)

        out_p = self.head_p(x).view(x.size(0), self.sg_num, 3)
        out_p = F.normalize(out_p, dim=-1, eps=1e-6)
        out_la = self.head_la(x).view(x.size(0), self.sg_num, 1)
        out_la = F.softplus(out_la) + self.min_positive
        out_w = self.head_w(x).view(x.size(0), self.sg_num, 1)
        out_w = F.softplus(out_w) + self.min_positive

        return out_p, out_la, out_w


class ASGNet(nn.Module):
    def __init__(
        self, asg_num=128, head_output_shapes=None, backbone_weights="DEFAULT"
    ):
        super().__init__()
        self.asg_num = asg_num
        self.min_positive = 1e-4

        densenet = models.densenet161(weights=densenet_weights(backbone_weights))
        self.encoder = nn.Sequential(*list(densenet.features.children()))

        self.decoder = nn.Sequential(
            nn.Conv2d(2208, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        in_features = 128 * 4 * 4
        head_features = 1024
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_features, head_features),
            nn.ReLU(inplace=True),
        )

        self.head_an = nn.Linear(head_features, asg_num)
        self.head_la = nn.Linear(head_features, asg_num)
        self.head_mu = nn.Linear(head_features, asg_num)
        self.head_w = nn.Linear(head_features, asg_num * 3)

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.feature_mlp(x)

        out_an = self.head_an(x).view(x.size(0), self.asg_num, 1)
        out_mu = self.head_mu(x).view(x.size(0), self.asg_num, 1)
        out_mu = F.softplus(out_mu) + self.min_positive
        out_la_delta = self.head_la(x).view(x.size(0), self.asg_num, 1)
        out_la = out_mu + F.softplus(out_la_delta)
        out_w = self.head_w(x).view(x.size(0), self.asg_num, 3)
        out_w = F.softplus(out_w) + self.min_positive

        return out_an, out_la, out_mu, out_w


class PanoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        if self.padding:
            padding = self.padding
            x = F.pad(x, (padding, padding, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, padding, padding), mode="reflect")
        return self.conv(x)


class HDRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            PanoConv(in_channels, out_channels),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            PanoConv(out_channels, out_channels),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class HDRDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True),
            HDRBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class HDRUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = HDRBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.block(torch.cat([x, skip], dim=1))


class HDRNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, sg_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.input_block = HDRBlock(n_channels, 64)
        self.down1 = HDRDown(64, 128)
        self.down2 = HDRDown(128, 256)
        self.sg_encoder = nn.Sequential(
            HDRBlock(sg_channels, 32),
            nn.MaxPool2d(2, ceil_mode=True),
            HDRBlock(32, 64),
            nn.MaxPool2d(2, ceil_mode=True),
            HDRBlock(64, 128),
        )
        self.condition_fusion = HDRBlock(384, 256)
        self.down3 = HDRDown(256, 512)
        self.down4 = HDRDown(512, 512)

        self.up1 = HDRUp(512, 512, 512)
        self.up2 = HDRUp(512, 256, 256)
        self.up3 = HDRUp(256, 128, 128)
        self.up4 = HDRUp(128, 64, 64)
        self.mask_attention = nn.Sequential(
            PanoConv(65, 64, bias=True),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            PanoConv(64 + sg_channels, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )

    def forward(self, x, sg, lum):
        sg_full = sg
        ldr_linear = srgb_to_linear(((x + 1) * 0.5).clamp(0, 1))
        inputs = torch.cat([ldr_linear * 2 - 1, lum], dim=1)

        x1 = self.input_block(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        sg_features = self.sg_encoder(sg)
        if sg_features.shape[-2:] != x3.shape[-2:]:
            sg_features = F.interpolate(
                sg_features,
                size=x3.shape[-2:],
                mode="nearest",
            )
        x3 = self.condition_fusion(torch.cat([x3, sg_features], dim=1))
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        features = self.up1(x5, x4)
        features = self.up2(features, x3)
        features = self.up3(features, x2)
        features = self.up4(features, x1)
        attention = self.mask_attention(torch.cat([features, lum], dim=1))
        if sg_full.shape[-2:] != features.shape[-2:]:
            sg_full = F.interpolate(
                sg_full,
                size=features.shape[-2:],
                mode="nearest",
            )
        conditioned = torch.cat([features * (1 + attention), sg_full], dim=1)
        residual = self.output(conditioned)
        return torch.log(ldr_linear.clamp_min(1e-8)) + residual
