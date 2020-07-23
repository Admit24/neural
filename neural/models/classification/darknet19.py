from collections import OrderedDict
from torch import nn


def darknet19(in_channels, num_classes):
    return Darknet19(in_channels, num_classes)


class Darknet19(nn.Sequential):

    def __init__(self, in_channels, num_classes, block_depth=[1, 3, 3, 5, 5]):

        def make_layer(in_channels, out_channels, num_blocks, use_pool=True):
            layers = [ConvBNReLU(in_channels, out_channels, 3, padding=1)]
            for i in range(1, num_blocks // 2):
                layers += [ConvBNReLU(out_channels, in_channels, 3, padding=1)]
                layers += [ConvBNReLU(in_channels, out_channels, 1)]
            if use_pool:
                layers += [nn.MaxPool2d(2)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', make_layer(in_channels, 32, 1)),
            ('layer1', make_layer(32, 64, block_depth[0])),
            ('layer3', make_layer(64, 128, block_depth[1])),
            ('layer4', make_layer(128, 256, block_depth[2])),
            ('layer4', make_layer(256, 512, block_depth[3])),
            ('layer5', make_layer(512, 1024, block_depth[4], use_pool=False)),
        ]))

        classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


def ConvBNReLU(in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )
