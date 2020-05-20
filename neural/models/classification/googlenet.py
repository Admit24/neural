from collections import OrderedDict
import torch
from torch import nn

__all__ = [
    'GoogleNet',
    'googlenet',
]


def googlenet(in_channels, out_channels):
    return GoogleNet(in_channels, out_channels)


class GoogleNet(nn.Sequential):

    def __init__(self, in_channels, num_channels):

        features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer2', nn.Sequential(
                ConvBlock(64, 64, kernel_size=1),
                ConvBlock(64, 192, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer3', nn.Sequential(
                InceptionBlock(192, 256, [64, (96, 128), (16, 32), 32]),
                InceptionBlock(256, 480, [128, (128, 192), (32, 96), 64]),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer4', nn.Sequential(
                InceptionBlock(480, 512, [192, (96, 208), (16, 48), 64]),
                InceptionBlock(512, 512, [160, (112, 224), (24, 64), 64]),
                InceptionBlock(512, 512, [128, (128, 256), (24, 64), 64]),
                InceptionBlock(512, 528, [112, (144, 288), (32, 64), 64]),
                InceptionBlock(528, 832, [256, (160, 320), (32, 128), 128]),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )),
            ('layer5', nn.Sequential(
                InceptionBlock(832, 832, [256, (160, 320), (32, 128), 128]),
                InceptionBlock(832, 1024, [384, (192, 384), (48, 128), 128]),
            ))
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(1024, num_channels),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, branch_channels):
        out_channels_ = sum([b[-1] if isinstance(b, tuple)
                             else b for b in branch_channels])
        if out_channels_ != out_channels:
            raise ValueError("the sum of the output channels of"
                             "each branch must match the output channels.")

        super().__init__()

        self.branch1 = ConvBlock(
            in_channels, branch_channels[0], kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, branch_channels[1][0], kernel_size=1),
            ConvBlock(branch_channels[1][0], branch_channels[1][1],
                      kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, branch_channels[2][0], kernel_size=1),
            ConvBlock(branch_channels[2][0], branch_channels[2][1],
                      kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, branch_channels[3], kernel_size=1),
        )

    def forward(self, input):
        return torch.cat([
            self.branch1(input),
            self.branch2(input),
            self.branch3(input),
            self.branch4(input),
        ], dim=1)


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels, eps=0.001),
        nn.ReLU(inplace=True),
    )
