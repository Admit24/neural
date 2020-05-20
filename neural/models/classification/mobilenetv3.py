from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from neural.nn import HardSwish
from neural.nn.functional import hard_sigmoid


def mobilenetv3_large(in_channels, num_classes):
    return MobileNetV3Large(in_channels, num_classes, width_multiplier=1.0)


def mobilenetv3_small(in_channels, num_classes):
    return MobileNetV3Small(in_channels, num_classes, width_multiplier=1.0)


class MobileNetV3Large(nn.Sequential):
    def __init__(self, in_channels, num_classes, width_multiplier=1):
        Block = InvertedResidualBlock

        def c(channels): return round_by(width_multiplier * channels)

        features = nn.Sequential(OrderedDict([
            ('head', nn.Sequential(
                ConvBlock(in_channels, c(16), 3,
                          padding=1, stride=2,
                          activation=HardSwish),
                Block(c(16), c(16), 3, 1, c(16))),
             ),
            ('layer1', nn.Sequential(
                Block(c(16), c(24), 3, 2, c(64)),
                Block(c(24), c(24), 3, 1, c(72)),
            )),
            ('layer2', nn.Sequential(
                Block(c(24), c(40), 5, 2, c(72), use_se=True),
                Block(c(40), c(40), 5, 1, c(120), use_se=True),
                Block(c(40), c(40), 5, 1, c(120), use_se=True),
            )),
            ('layer3', nn.Sequential(
                Block(c(40), c(80), 3, 2, c(240), HardSwish, False),
                Block(c(80), c(80), 3, 1, c(200), HardSwish, False),
                Block(c(80), c(80), 3, 1, c(184), HardSwish, False),
                Block(c(80), c(80), 3, 1, c(184), HardSwish, False),
                Block(c(80), c(112), 3, 1, c(480), HardSwish, True),
                Block(c(112), c(112), 3, 1, c(672), HardSwish, True),
            )),
            ('layer4', nn.Sequential(
                Block(c(112), c(160), 5, 2, c(672), HardSwish, True),
                Block(c(160), c(160), 5, 1, c(960), HardSwish, True),
                Block(c(160), c(160), 5, 1, c(960), HardSwish, True),
            )),
            ('tail', ConvBlock(c(160), c(960), 1, activation=HardSwish)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(c(960), c(1280)),
            HardSwish(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(c(1280), num_classes),
        )

        super(MobileNetV3Large, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class MobileNetV3Small(nn.Sequential):
    def __init__(self, in_channels, num_classes, width_multiplier=1):
        Block = InvertedResidualBlock

        def c(channels): return round_by(width_multiplier * channels)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, c(16), 3,
                               padding=1, stride=2,
                               activation=HardSwish)),
            ('layer1', nn.Sequential(
                Block(c(16), c(16), 3, 2, c(16), use_se=True),
            )),
            ('layer2', nn.Sequential(
                Block(c(16), c(24), 3, 2, c(72)),
                Block(c(24), c(24), 3, 1, c(88)),
            )),
            ('layer3', nn.Sequential(
                Block(c(24), c(40), 5, 2, c(96), HardSwish, True),
                Block(c(40), c(40), 5, 1, c(240), HardSwish, True),
                Block(c(40), c(40), 5, 1, c(240), HardSwish, True),
                Block(c(40), c(48), 5, 1, c(120), HardSwish, True),
                Block(c(48), c(48), 5, 1, c(144), HardSwish, True),
            )),
            ('layer4', nn.Sequential(
                Block(c(48), c(96), 5, 2, c(288), HardSwish, True),
                Block(c(96), c(96), 5, 2, c(576), HardSwish, True),
                Block(c(96), c(96), 5, 2, c(576), HardSwish, True),
            )),
            ('tail', ConvBlock(c(96), c(576), 1, activation=HardSwish)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(c(576), c(1024)),
            HardSwish(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(c(1024), num_classes),
        )

        super(MobileNetV3Small, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expansion_channels, activation=nn.ReLU, use_se=False):
        super(InvertedResidualBlock, self).__init__()

        self.expansion = (
            ConvBlock(in_channels, expansion_channels, 1,
                      activation=activation)
            if expansion_channels != out_channels
            else None
        )

        self.conv = ConvBlock(
            expansion_channels, expansion_channels, kernel_size,
            padding=kernel_size // 2, stride=stride,
            groups=expansion_channels, activation=activation)

        self.se = (
            SEBlock(expansion_channels, expansion_channels, reduction_ratio=4)
            if use_se else None
        )

        self.reduction = ConvBlock(expansion_channels, out_channels, 1,
                                   activation=None)

    def forward(self, input):
        x = self.expansion(input) if self.expansion is not None else input
        x = self.conv(x)
        x = self.se(x) if self.se is not None else x
        x = self.reduction(x)
        if input.shape == x.shape:
            x = x + input
        return x


class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()

        red_channels = round_by(in_channels / reduction_ratio, 8)
        self.conv1 = nn.Conv2d(in_channels, red_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(red_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return input * hard_sigmoid(x)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, groups=1, activation=nn.ReLU):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if activation is not None:
        layers += [activation(inplace=True)]
    return nn.Sequential(*layers)


def round_by(channels, divisor=8):
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
