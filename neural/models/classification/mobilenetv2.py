from collections import OrderedDict
from torch import nn
from neural.utils.hub import configure_model

__all__ = [
    'MobileNetV2',
    'mobilenetv2',
]


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/mobilenetv2/mobilenetv2_1_0-8b85393a.pth',
    },
})
def mobilenetv2(in_channels=3, num_classes=1000, width_multiplier=1):
    return MobileNetV2(in_channels, num_classes,
                       width_multiplier=width_multiplier)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/mobilenetv2/mobilenetv2_0_75-b955620c.pth',
    }
})
def mobilenetv2_0_75(in_channels=3, num_classes=1000):
    return mobilenetv2(in_channels, num_classes, width_multiplier=0.75)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/mobilenetv2/mobilenetv2_0_5-bd1d2a68.pth',
    }
})
def mobilenetv2_0_5(in_channels=3, num_classes=1000):
    return mobilenetv2(in_channels, num_classes, width_multiplier=0.5)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/mobilenetv2/mobilenetv2_0.35-cc0f5647.pth',
    }
})
def mobilenetv2_0_35(in_channels=3, num_classes=1000):
    return mobilenetv2(in_channels, num_classes, width_multiplier=0.35)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/mobilenetv2/mobilenetv2_0.25-e542a5f2.pth',
    }
})
def mobilenetv2_0_25(in_channels=3, num_classes=1000):
    return mobilenetv2(in_channels, num_classes, width_multiplier=0.25)


class MobileNetV2(nn.Sequential):

    def __init__(self, in_channels, num_classes, width_multiplier=1.0):

        def c(channels): return round_by(width_multiplier * channels)

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

        out_channels = max(1280, c(1280))

        features = nn.Sequential(OrderedDict([
            ('head', ConvBlock(in_channels, c(32), 3, padding=1, stride=2)),
            ('layer1', make_layer(c(32), c(16), expansion=1)),
            ('layer2', make_layer(c(16), c(24), num_blocks=2, stride=2)),
            ('layer3', make_layer(c(24), c(32), num_blocks=3, stride=2)),
            ('layer4', make_layer(c(32), c(64), num_blocks=4, stride=2)),
            ('layer5', make_layer(c(64), c(96), num_blocks=3, stride=1)),
            ('layer6', make_layer(c(96), c(160), num_blocks=3, stride=2)),
            ('layer7', make_layer(c(160), c(320), num_blocks=1, stride=1)),
            ('tail', ConvBlock(c(320), out_channels, 1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes),
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
            if expansion != 1 else nn.Identity())
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


def round_by(channels, divisor=8):
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
