from collections import OrderedDict
from torch import nn

__all__ = [
    'Vgg',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
]


def vgg11(in_channels=3, num_classes=1000):
    return Vgg(in_channels, num_classes, block_depth=[1, 1, 2, 2, 2])


def vgg13(in_channels=3, num_classes=1000):
    return Vgg(in_channels, num_classes, block_depth=[2, 2, 2, 2, 2])


def vgg16(in_channels=3, num_classes=1000):
    return Vgg(in_channels, num_classes, block_depth=[2, 2, 3, 3, 3])


def vgg19(in_channels=3, num_classes=1000):
    return Vgg(in_channels, num_classes, block_depth=[2, 2, 4, 4, 4])


class Vgg(nn.Sequential):

    def __init__(self, in_channels, num_classes, block_depth):

        features = nn.Sequential(OrderedDict([
            ('layer1', VggLayer(in_channels, 64, block_depth[0])),
            ('layer2', VggLayer(64, 128, block_depth[1])),
            ('layer3', VggLayer(128, 256, block_depth[2])),
            ('layer4', VggLayer(256, 512, block_depth[3])),
            ('layer5', VggLayer(512, 512, block_depth[4])),
        ]))

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


def VggLayer(in_channels, out_channels, num_blocks):
    layers = []
    layers.append(ConvBlock(in_channels, out_channels))
    for _ in range(1, num_blocks):
        layers.append(ConvBlock(out_channels, out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
