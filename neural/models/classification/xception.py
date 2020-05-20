from collections import OrderedDict
from torch import nn


class Xception(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        features = nn.Sequential(OrderedDict([
            ('head', nn.Sequential(
                ConvBlock(in_channels, 32, 3, padding=1, stride=2),
                ConvBlock(32, 64, 3, padding=1),
            )),
            ('layer1', ResidualBlock(64, 128, stride=2, num_blocks=2,
                                     first_relu=False)),
            ('layer2', ResidualBlock(128, 256, stride=2, num_blocks=2)),
            ('layer3', ResidualBlock(256, 728, stride=2, num_blocks=2)),
            ('layer4', nn.Sequential(*[
                ResidualBlock(728, 728)
                for _ in range(8)
            ])),
            ('layer5', nn.Sequential(
                ResidualBlock(728, 1024, stride=2,
                              num_blocks=2, grow_first=False),
                nn.Sequential(
                    SeparableConvBlock(1024, 1536, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    SeparableConvBlock(1536, 2048, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
            )),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(2048, num_classes),
        )

        super(Xception, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 num_blocks=3, grow_first=True, first_relu=True):
        super(ResidualBlock, self).__init__()

        first_conv = []
        if first_relu:
            first_conv += [nn.ReLU(inplace=True)]
        first_conv += [SeparableConvBlock(in_channels, out_channels, 3,
                                          padding=1, use_relu=False)]
        self.conv1 = nn.Sequential(*first_conv)

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConvBlock(out_channels, out_channels, 3,
                               padding=1, use_relu=False),
        )

        if num_blocks == 3:
            self.conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                SeparableConvBlock(out_channels, out_channels,
                                   3, padding=1, use_relu=False),
            )
        else:
            self.conv3 = None

        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.pool = None
            self.skip = None

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        if self.conv3 is not None:
            x = self.conv3(x)
        if self.pool is not None:
            x = self.pool(x)

        if self.skip is not None:
            input = self.skip(input)

        return x + input


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


def SeparableConvBlock(in_channels, out_channels, kernel_size,
                       padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
