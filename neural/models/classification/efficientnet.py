from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch.nn import functional as F
from neural.nn import Swish
from neural.utils.hub import configure_model


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b0-imagenet-e78091d2.pth',
    }
})
def efficientnet_b0(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.0,
                        depth_multiplier=1.0,
                        dropout_rate=0.2)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b1-imagenet-28855b93.pth'
    }
})
def efficientnet_b1(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.0,
                        depth_multiplier=1.1,
                        dropout_rate=0.2)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b2-imagenet-8bab1f6c.pth',
    }
})
def efficientnet_b2(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.1,
                        depth_multiplier=1.2,
                        dropout_rate=0.3)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b3-imagenet-0b7b4186.pth',
    }
})
def efficientnet_b3(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.2,
                        depth_multiplier=1.4,
                        dropout_rate=0.3)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b4-imagenet-79852444.pth',
    }
})
def efficientnet_b4(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.4,
                        depth_multiplier=1.8,
                        dropout_rate=0.4)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b5-imagenet-4cca5e55.pth',
    }
})
def efficientnet_b5(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.6,
                        depth_multiplier=2.2,
                        dropout_rate=0.4)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b6-imagenet-ba439be8.pth',
    }
})
def efficientnet_b6(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=1.8,
                        depth_multiplier=2.6,
                        dropout_rate=0.5)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/efficientnet/efficientnet_b7-imagenet-5218c83a.pth',
    }
})
def efficientnet_b7(in_channels, out_channels):
    return EfficientNet(in_channels, out_channels,
                        width_multiplier=2.0,
                        depth_multiplier=3.1,
                        dropout_rate=0.5)


class EfficientNet(nn.Sequential):

    def __init__(self, in_channels, num_classes,
                 width_multiplier=1,
                 depth_multiplier=1,
                 dropout_rate=0.2):

        def c(channels): return round_by(width_multiplier * channels)

        def d(depth): return ceil(depth * depth_multiplier)

        def make_layer(in_channels, out_channels, num_blocks=1,
                       kernel_size=3, stride=1, expansion_ratio=6):
            layers = [MBConvBlock(
                in_channels, out_channels, kernel_size,
                stride=stride, expansion_ratio=expansion_ratio)]
            for _ in range(1, num_blocks):
                layers += [
                    MBConvBlock(out_channels, out_channels, kernel_size,
                                expansion_ratio=expansion_ratio)
                ]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, c(32), 3, padding=1, stride=2)),
            ('layer1', make_layer(c(32), c(16), d(1), expansion_ratio=1)),
            ('layer2', make_layer(c(16), c(24), d(2), stride=2)),
            ('layer3', make_layer(c(24), c(40), d(2),
                                  kernel_size=5, stride=2)),
            ('layer4', make_layer(c(40), c(80), d(3), stride=2)),
            ('layer5', make_layer(c(80), c(112), d(3), kernel_size=5)),
            ('layer6', make_layer(c(112), c(192), d(4),
                                  kernel_size=5, stride=2)),
            ('layer7', make_layer(c(192), c(320), d(1))),
            ('tail', ConvBlock(c(320), c(1280), 1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(c(1280), num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 expansion_ratio=6, reduction_ratio=4,
                 dropout_p=0.2):
        super().__init__()

        hidden_channels = in_channels * expansion_ratio

        self.expansion = (
            ConvBlock(in_channels, hidden_channels, 1)
            if expansion_ratio != 1
            else None
        )

        self.conv1 = ConvBlock(
            hidden_channels, hidden_channels, kernel_size,
            padding=kernel_size // 2, stride=stride,
            groups=hidden_channels)

        self.conv2 = SEBlock(hidden_channels, hidden_channels,
                             reduction_ratio=reduction_ratio * expansion_ratio)

        self.conv3 = ConvBlock(
            hidden_channels, out_channels, 1,
            use_activation=False)

        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = input if self.expansion is None else self.expansion(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if input.shape == x.shape:
            x = self.dropout(x)
            return input + x
        else:
            return x


class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()

        red_channels = in_channels // reduction_ratio
        self.conv1 = nn.Conv2d(in_channels, red_channels, 1)
        self.activation = Swish()
        self.conv2 = nn.Conv2d(red_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return input * torch.sigmoid(x)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, groups=1,
              use_activation=True):

    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
    ]
    if use_activation:
        layers += [Swish()]
    return nn.Sequential(*layers)


def round_by(channels, divisor=8):
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
