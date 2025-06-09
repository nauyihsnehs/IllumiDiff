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
            # nn.BatchNorm2d(out_channels),
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

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) if conv_num == 2 else SingleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) if conv_num == 2 else SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
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
        # Load the pretrained Densenet161 and extract the features layer
        # densenet = models.resnet18(pretrained=True)
        # densenet.fc = nn.Identity()
        # densenet.avgpool = nn.Identity()
        # self.encoder = densenet

        densenet = models.densenet161(pretrained=True)
        self.encoder = densenet.features

        # densenet = models.vit_b_16(pretrained=True)
        # densenet.heads = nn.Identity()
        # self.encoder = densenet

        # densenet = models.mobilenet_v3_large(pretrained=True)
        # self.encoder = densenet.features
        # self.encoder = nn.Sequential(
        #     # Layer 1: capture edges and gradients
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),  # Downsample
        #
        #     # Layer 2: texture and low-level lighting
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #
        #     # Layer 3: mid-level abstraction (e.g., shadow zones)
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #
        #     # Layer 4: highlight/shadow separation
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #
        #     # Layer 5: lighting distribution pattern (global context)
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #
        #     # Layer 5: lighting distribution pattern (global context)
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1024),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2),
        # )

        # Auxiliary path for processing the mask
        self.mask_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adjust the decoder conv layer input channels
        self.decoder_conv = nn.Sequential(
            # nn.Conv2d(512 + 128, 512, kernel_size=3, padding=1),
            nn.Conv2d(2208 + 128, 512, kernel_size=3, padding=1),
            # nn.Conv2d(2048 + 128, 512, kernel_size=3, padding=1),
            # 2208 is DenseNet-161's last feature layer output channels
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

        # Fully connected heads for each output
        # self.heads = nn.ModuleList()
        # for i in range(n_heads):
        #     self.heads.append(
        #         nn.Sequential(nn.Flatten(), nn.Linear(256 * (256 // 32) * (256 // 32), a * head_channels[i]), nn.Unfold((1, a)) if head_channels[i] != a else nn.Identity()))
        in_features = 128 * 1 * 1
        self.head_p = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 3))
        self.head_la = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 1))
        self.head_w = nn.Sequential(nn.Flatten(), nn.Linear(in_features, sg_num * 3))

    def forward(self, x, mask):
        # Process the mask
        mask_features = self.mask_processor(mask)

        # Pass through the encoder
        x = self.encoder(x)  # [:,1:].reshape(1, -1, 8, 8)

        # Resize mask features to match the encoder output's spatial dimensions
        mask_features = nn.functional.interpolate(mask_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        # Concatenate mask features with the encoder's output
        x = torch.cat([x, mask_features], dim=1)

        # Pass through decoder conv layers
        x = self.decoder_conv(x)

        # Generate outputs from heads
        out_p = self.head_p(x).view(x.size(0), self.sg_num, 3)
        out_p = out_p / out_p.norm(dim=-1, keepdim=True)
        out_la = self.head_la(x).view(x.size(0), self.sg_num, 1)
        out_la = 1 / (torch.sigmoid(out_la) + 1e-5)
        out_w = self.head_w(x).view(x.size(0), self.sg_num, 3)
        out_w = 1 / (torch.sigmoid(out_w) + 1e-5)

        # return torch.cat((out_p, out_la, out_w), dim=-1)
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
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1)

        # Apply each head to the flattened input
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
        # self.down3 = (Down(256, 512))
        self.down3 = (Down(512, 1024))
        self.down4 = (Down(1024, 1024))
        self.up1 = (Up(2048, 512))
        self.up2 = (Up(1024, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))
        # Auxiliary path for processing the mask
        self.sg_en = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.mask_en = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
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


if __name__ == '__main__':
    import torch.utils.benchmark as benchmark

    model = None
    # 创建一个基准对象
    timer = benchmark.Timer(
        stmt='model(*input)',  # 被测代码
        # setup='input = torch.randn(1, 3, 256, 512).cuda(); model = IDNet(3, 1).cuda()',  # 环境准备
        # setup='input = [torch.randn(1, 3, 240, 320).cuda(),torch.randn(1, 1, 240, 320).cuda()]; model = SGNet().cuda()',  # 环境准备
        # setup='input = [torch.randn(1, 3, 256, 256).cuda(),torch.randn(1, 1, 256, 256).cuda()]; model = SGNet().cuda()',  # 环境准备
        # setup='input = [torch.randn(1, 3, 224, 224).cuda(),torch.randn(1, 1, 224, 224).cuda()]; model = SGNet().cuda()',  # 环境准备
        # setup='input = torch.randn(1, 3, 256, 256).cuda(); model = ASGNet().cuda()',  # 环境准备
        # setup='input = [torch.randn(1, 3, 256, 512).cuda(),torch.randn(1, 1, 256, 512).cuda(),torch.randn(1, 1, 256, 512).cuda()]; model = HDRNet(4, 3).cuda()',  # 环境准备
        setup='input = [torch.randn(1, 3, 128, 256).cuda(),torch.randn(1, 1, 128, 256).cuda(),torch.randn(1, 1, 128, 256).cuda()]; model = HDRNet(4, 3).cuda()',  # 环境准备
        globals={'model': model, 'IDNet': IDNet, 'SGNet': SGNet, 'ASGNet': ASGNet, 'HDRNet': HDRNet}  # 全局变量
    )

    # 进行多次测量并获取平均时间
    print(timer.timeit(1000))  # 100次重复执行
    # model = None
    # timer = benchmark.Timer(
    #     stmt='model(input_data)',  # 你想测试的代码
    #     setup='input_data = torch.randn(64, 1024).cuda(); model = torch.nn.Linear(1024, 1024).cuda()',  # 设置测试环境
    #     globals={'model': model}  # 定义全局变量
    # )
    #
    # # 进行 100 次重复执行并计算平均时间
    # print(timer.timeit(100))  # 执行 100 次并输出结果
