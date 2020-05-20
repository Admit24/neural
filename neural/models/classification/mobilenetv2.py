from collections import OrderedDict

from torch import nn

__all__ = [
    'MobileNetV2',
    'mobilenetv2', 'mobilenetv2_0_5x', 'mobilenetv2_2x',
]


def mobilenetv2(in_channels, num_classes):
    return MobileNetV2(in_channels, num_classes)


def mobilenetv2_0_5x(in_channels, num_classes):
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.5)


def mobilenetv2_2x(in_channels, num_classes):
    return MobileNetV2(in_channels, num_classes, width_multiplier=2.0)


class MobileNetV2(nn.Sequential):

    def __init__(self, in_channels, num_classes, width_multiplier=1.0):

        def c(channels): return int(width_multiplier * channels)

        def make_layer(in_channels, out_channels, num_blocks=1,
                       expansion=6, stride=1):
            layers = [InvertedResidualBlock(
                in_channels, out_channels,
                stride=stride, expansion=expansion)]
            for _ in range(1, num_blocks):
                layers += [InvertedResidualBlock(
                    out_channels, out_channels,
                    stride=1, expansion=expansion)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, c(32), 3, padding=1, stride=2)),
            ('layer1', make_layer(c(32), c(16), expansion=1)),
            ('layer2', make_layer(c(16), c(24), num_blocks=2, stride=2)),
            ('layer3', make_layer(c(24), c(32), num_blocks=3, stride=2)),
            ('layer4', make_layer(c(32), c(64), num_blocks=4, stride=2)),
            ('layer5', make_layer(c(64), c(96), num_blocks=3, stride=1)),
            ('layer6', make_layer(c(96), c(160), num_blocks=3, stride=2)),
            ('layer7', make_layer(c(160), c(320), num_blocks=1, stride=1)),
            ('tail', ConvBlock(c(320), c(1280), 3, padding=1, stride=1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(c(1280), num_classes),
        )

        super(MobileNetV2, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion=6):
        super().__init__()

        hidden_channels = in_channels * expansion
        self.conv1 = (
            ConvBlock(in_channels, hidden_channels, 1)
            if expansion != 1 else nn.Sequential())
        self.conv2 = DWConvBlock(
            hidden_channels, hidden_channels, 3,
            padding=1, stride=stride)
        self.conv3 = ConvBlock(
            hidden_channels, out_channels, 1, use_relu=False)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if input.shape == x.shape:
            x = input + x
        return x


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)


def DWConvBlock(in_channels, out_channels, kernel_size,
                padding=0, stride=1, use_relu=True):
    if in_channels != out_channels:
        raise ValueError(
            "input channels must be the same as the"
            "output channels in the depthwise convolution.")

    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)
