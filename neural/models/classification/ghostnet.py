from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from math import ceil
from neural.nn import hard_sigmoid
from neural.utils.hub import configure_model


@configure_model({
    'imagenet': {
        'in_channels': 3, 'out_channels': 1000,
        'state_dict': 'http://files.deeplar.tk/neural/weights/ghostnet/ghostnet_1_0-imagenet-565e8ab8.pth',
    },
})
def ghostnet_1_0(in_channels, out_channels):
    return GhostNet(in_channels, out_channels)


class GhostNet(nn.Sequential):

    def __init__(self, in_channels, num_classes, width_multiplier=1.0):

        def c(channels): return round_by(width_multiplier * channels)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, c(16), 3, padding=1, stride=2)),
            ('layer1', nn.Sequential(
                GhostBottleneck(c(16), c(16), c(16)),
                GhostBottleneck(c(16), c(24), c(48), stride=2),
            )),
            ('layer2', nn.Sequential(
                GhostBottleneck(c(24), c(24), c(72)),
                GhostBottleneck(c(24), c(40), c(72), kernel_size=5, stride=2, use_se=True),
            )),
            ('layer3', nn.Sequential(
                GhostBottleneck(c(40), c(40), c(120), kernel_size=5, use_se=True),
                GhostBottleneck(c(40), c(80), c(240), stride=2),
            )),
            ('layer4', nn.Sequential(
                GhostBottleneck(c(80), c(80), c(200)),
                GhostBottleneck(c(80), c(80), c(184)),
                GhostBottleneck(c(80), c(80), c(184)),
                GhostBottleneck(c(80), c(112), c(480), use_se=True),
                GhostBottleneck(c(112), c(112), c(672), use_se=True),
                GhostBottleneck(c(112), c(160), c(672), kernel_size=5, stride=2, use_se=True),
            )),
            ('layer5', nn.Sequential(
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, ),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, use_se=True),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, ),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, use_se=True),
                ConvBlock(c(160), c(960), 1),
            )),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(c(960), c(1280), 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(c(1280), num_classes),
        )

        super(GhostNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_channels,  kernel_size=3, stride=1, expansion_ratio=4, use_se=False):
        super(GhostBottleneck, self).__init__()

        self.conv1 = GhostModule(in_channels, expansion_channels)
        self.conv2 = (ConvBlock(expansion_channels, expansion_channels, kernel_size, padding=kernel_size // 2, stride=stride, groups=expansion_channels, use_relu=False)
                      if stride != 1 else nn.Identity())
        self.se = SEBlock(expansion_channels, expansion_channels) if use_se else nn.Identity()
        self.conv3 = GhostModule(expansion_channels, out_channels, use_relu=False)

        if stride == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                ConvBlock(in_channels, in_channels, kernel_size,
                          padding=kernel_size//2, stride=stride,
                          groups=in_channels, use_relu=False),
                ConvBlock(in_channels, out_channels, 1, use_relu=False),
            )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            input = self.downsample(input)
        return x + input


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(1, 3), stride=1, reduction_ratio=2, use_relu=True):
        super(GhostModule, self).__init__()

        init_channels = ceil(out_channels / reduction_ratio)
        new_channels = init_channels * (reduction_ratio - 1)

        self.conv1 = ConvBlock(in_channels, init_channels, kernel_sizes[0], stride=stride,
                               padding=kernel_sizes[0]//2, use_relu=use_relu)
        self.conv2 = ConvBlock(init_channels, new_channels, kernel_sizes[1],
                               padding=kernel_sizes[1] // 2,
                               groups=init_channels,
                               use_relu=use_relu)

        self.out_channels = out_channels

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)
        return x[:, :self.out_channels, ...]


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()

        reduced_channels = round_by(in_channels / reduction_ratio, 4)

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return input * hard_sigmoid(x)


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def round_by(channels, divisor=8):
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
