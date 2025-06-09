import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_num=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels) if conv_num == 2 else SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, conv_num=2):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) if conv_num == 2 else SingleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) if conv_num == 2 else SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IDNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(IDNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 64))
        self.down1 = (Down(64, 128, conv_num=1))
        self.down2 = (Down(128, 256, conv_num=1))
        self.down3 = (Down(256, 512, conv_num=1))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, conv_num=1))
        self.up1 = (Up(1024, 512 // factor, bilinear, conv_num=1))
        self.up2 = (Up(512, 256 // factor, bilinear, conv_num=1))
        self.up3 = (Up(256, 128 // factor, bilinear, conv_num=1))
        self.up4 = (Up(128, 64, bilinear, conv_num=1))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SGNet(nn.Module):
    def __init__(self, sg_num=12):
        super(SGNet, self).__init__()

        self.sg_num = sg_num

        densenet = models.densenet161(pretrained=True)
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
            nn.AdaptiveAvgPool2d(1)
        )

        in_features = 128 * 1 * 1
        self.head_p = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 3))
        self.head_la = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 1))
        self.head_w = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 3))

    def forward(self, x, mask):
        mask_features = self.mask_processor(mask)
        x = self.encoder(x)  # [:,1:].reshape(1, -1, 8, 8)
        mask_features = nn.functional.interpolate(mask_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, mask_features], dim=1)
        x = self.decoder_conv(x)

        out_p = self.head_p(x).view(x.size(0), self.sg_num, 3)
        out_p = out_p / out_p.norm(dim=-1, keepdim=True)
        out_la = self.head_la(x).view(x.size(0), self.sg_num, 1)
        out_la = 1 / (torch.sigmoid(out_la) + 1e-5)
        out_w = self.head_w(x).view(x.size(0), self.sg_num, 3)
        out_w = 1 / (torch.sigmoid(out_w) + 1e-5)

        return out_p, out_la, out_w


class ASGNet(nn.Module):
    def __init__(self, asg_num=128, num_heads=4, head_output_shapes=None):
        super(ASGNet, self).__init__()
        self.asg_num = asg_num

        if head_output_shapes is None:
            self.head_output_shapes = [(asg_num, 1), (asg_num, 1), (asg_num, 1), (asg_num, 3)]
        densenet = models.densenet161(pretrained=True)
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
            nn.AdaptiveAvgPool2d(2)
        )

        in_features = 128 * 2 * 2

        self.head_an = nn.Sequential(nn.Linear(in_features, asg_num * 1), nn.Sigmoid())
        self.head_la = nn.Sequential(nn.Linear(in_features, asg_num * 1))
        self.head_mu = nn.Sequential(nn.Linear(in_features, asg_num * 1))
        self.head_w = nn.Sequential(nn.Linear(in_features, asg_num * 3))

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1)

        out_an = self.head_an(x).view(x.size(0), self.asg_num, 1)
        out_la = self.head_la(x).view(x.size(0), self.asg_num, 1)
        out_la = torch.abs(out_la)
        out_mu = self.head_mu(x).view(x.size(0), self.asg_num, 1)
        out_mu = torch.abs(out_mu)
        out_w = self.head_w(x).view(x.size(0), self.asg_num, 3)
        out_w = torch.abs(out_w)

        return out_an, out_la, out_mu, out_w


class HDRNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(HDRNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(512, 1024))
        self.down4 = (Down(1024, 1024))
        self.up1 = (Up(2048, 512))
        self.up2 = (Up(1024, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))
        self.sg_en = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask_en = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, sg, mask):
        inputs = torch.cat([x, mask], dim=1)
        sg = self.sg_en(sg)
        mask = self.mask_en(mask)
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = torch.cat([x3, sg], dim=1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = x * mask
        x = self.up4(x, x1)
        logits = torch.abs(self.outc(x))
        return logits

