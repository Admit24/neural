from collections import OrderedDict
from torch import nn


__all__ = [
    'RegNet',
]


def regnet_002(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 1, 4, 7], [24, 56, 152, 368], 8)


def regnet_004(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 2, 7, 12], [32, 64, 160, 384], 16)


def regnet_006(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 3, 5, 7], [48, 96, 240, 528], 24)


def regnet_008(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 3, 7, 5], [64, 128, 288, 672], 16)


def regnet_016(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 4, 10, 2], [72, 168, 408, 912], group_width=24)


def regnet_032(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48)


def regnet_040(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 14, 2], [80, 240, 560, 1360], group_width=40)


def regnet_064(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 4, 10, 1], [168, 392, 784, 1624], group_width=56)


def regnet_080(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120)


def regnet_120(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112)


def regnet_160(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 6, 13, 1], [256, 512, 896, 2048], group_width=128)


def regnet_320(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 7, 13, 1], [336, 672, 1344, 2520], group_width=168)


class RegNet(nn.Sequential):
    def __init__(self, in_channels, num_classes,
                 block_depth, block_channels, group_width=1):

        def make_layer(in_channels, out_channels, num_blocks, stride=2):
            layers = [BottleneckBlock(in_channels, out_channels, stride=stride, group_width=group_width)]
            for _ in range(1, num_blocks):
                layers += [BottleneckBlock(out_channels, out_channels, group_width=group_width)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, 32, 3, padding=1, stride=2)),
            ('layer1', make_layer(32, block_channels[0], block_depth[0])),
            ('layer2', make_layer(block_channels[0], block_channels[1], block_depth[1])),
            ('layer3', make_layer(block_channels[1], block_channels[2], block_depth[2])),
            ('layer4', make_layer(block_channels[2], block_channels[3], block_depth[3])),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(block_channels[-1], num_classes),
        )

        super(RegNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group_width=1, expansion=1):
        super(BottleneckBlock, self).__init__()

        mid_channels = int(expansion * out_channels)

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(mid_channels, mid_channels, 3, padding=1, stride=stride,
                               groups=mid_channels // min(mid_channels, group_width))
        self.conv3 = ConvBlock(mid_channels, out_channels, 1, use_relu=False)

        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvBlock(in_channels, out_channels, 1, stride=stride)
        else:
            self.downsample = None

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        input = input if self.downsample is None else self.downsample(input)
        return self.activation(x + input)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, groups=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
