import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ContextNet',
    'contextnet12',
    'contextnet14',
    'contextnet18',
]


def contextnet12(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=2)


def contextnet14(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=4)


def contextnet18(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=8)


class ContextNet(nn.Module):
    scale_factor: int = 4

    def __init__(self, in_channels, out_channels,
                 scale_factor=4,
                 width_multiplier=1):
        super(ContextNet, self).__init__()

        self.scale_factor = scale_factor

        def c(channels): return int(channels * width_multiplier)

        self.spatial = nn.Sequential(
            ConvBlock(in_channels, c(32), 3, padding=1, stride=2),
            DWConvBlock(c(32), c(32), kernel_size=3, padding=1, stride=2),
            ConvBlock(c(32), c(64), 1),
            DWConvBlock(c(64), c(64), kernel_size=3, padding=1, stride=2),
            ConvBlock(c(64), c(128), 1),
            DWConvBlock(c(128), c(128), kernel_size=3, padding=1, stride=1),
            ConvBlock(c(128), c(128), 1),
        )

        self.context = nn.Sequential(
            ConvBlock(in_channels, c(32), 3, padding=1, stride=2),
            BottleneckBlock(c(32), c(32), expansion=1),
            BottleneckBlock(c(32), c(32), expansion=6),
            LinearBottleneck(c(32), c(48), 3, stride=2),
            LinearBottleneck(c(48), c(64), 3, stride=2),
            LinearBottleneck(c(64), c(96), 2),
            LinearBottleneck(c(96), c(128), 2),
            ConvBlock(c(128), c(128), 3, padding=1),
        )

        self.feature_fusion = FeatureFusionModule((c(128), c(128)), c(128))

        self.classifier = Classifier(c(128), out_channels)

    def forward(self, input):
        spatial = self.spatial(input)

        context = F.interpolate(
            input, scale_factor=1 / self.scale_factor,
            mode='bilinear', align_corners=True)
        context = self.context(context)

        fusion = self.feature_fusion(context, spatial)

        classes = self.classifier(fusion)

        return F.interpolate(
            classes, scale_factor=8,
            mode='bilinear', align_corners=True)


def Classifier(in_channels, out_channels):
    return nn.Sequential(
        DWConvBlock(in_channels, in_channels, 3, padding=1),
        ConvBlock(in_channels, in_channels, 1),
        DWConvBlock(in_channels, in_channels, 3, padding=1),
        ConvBlock(in_channels, in_channels, 1),
        nn.Dropout(p=0.1),
        nn.Conv2d(in_channels, out_channels, 1),
    )


def LinearBottleneck(in_channels, out_channels, num_blocks,
                     expansion=6, stride=1):
    layers = [
        BottleneckBlock(
            in_channels, out_channels,
            stride=stride, expansion=expansion)]

    for _ in range(1, num_blocks):
        layers += [
            BottleneckBlock(
                out_channels, out_channels, expansion=expansion)
        ]
    return nn.Sequential(*layers)


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        lowres_channels, highres_channels = in_channels
        self.lowres = nn.Sequential(
            DWConvBlock(lowres_channels, lowres_channels,
                        kernel_size=3, padding=4, dilation=4),
            ConvBlock(lowres_channels, out_channels, 1, use_relu=False)
        )
        self.highres = ConvBlock(
            highres_channels, out_channels, 1, use_relu=False)

    def forward(self, lowres, highres):
        lowres = F.interpolate(
            lowres, size=highres.shape[2:],
            mode='bilinear', align_corners=True)
        lowres = self.lowres(lowres)

        highres = self.highres(highres)

        return F.relu(lowres + highres)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(BottleneckBlock, self).__init__()

        expansion_channels = in_channels * expansion
        self.conv1 = ConvBlock(in_channels, expansion_channels, 1)
        self.conv2 = DWConvBlock(
            expansion_channels, expansion_channels, 3,
            padding=1, stride=stride)
        self.conv3 = ConvBlock(
            expansion_channels, out_channels, 1, use_relu=False)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if x.shape == input.shape:
            x = input + x
        return F.relu(x)


def DWConvBlock(in_channels, out_channels, kernel_size,
                padding=0, stride=1, dilation=1,
                use_relu=True):
    if in_channels != out_channels:
        raise ValueError(
            "input and output channels must be the same in depthwise convolution")

    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, dilation=dilation,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
