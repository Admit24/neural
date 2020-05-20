from collections import OrderedDict
from functools import partial
from torch import nn
from torch.nn import functional as F


def drn_a_18(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'A', [2, 2, 2, 2], BasicBlock)


def drn_a_34(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'A', [3, 4, 6, 3], BasicBlock)


def drn_a_50(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'A', [3, 4, 6, 3],
               partial(Bottleneck, expansion=4), expansion=4)


def drn_c_26(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'C', [2, 2, 2, 2, 1, 1], BasicBlock)


def drn_c_42(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'C', [3, 4, 6, 3, 1, 1], BasicBlock)


def drn_c_58(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'C', [3, 4, 6, 3, 1, 1],
               partial(Bottleneck, expansion=4), expansion=4)


def drn_d_22(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [2, 2, 2, 2, 1, 1], BasicBlock)


def drn_d_24(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [2, 2, 2, 2, 2, 2], BasicBlock)


def drn_d_38(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 6, 3, 1, 1], BasicBlock)


def drn_d_40(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 6, 3, 2, 2], BasicBlock)


def drn_d_54(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 6, 3, 1, 1],
               partial(Bottleneck, expansion=4), expansion=4)


def drn_d_56(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 6, 3, 2, 2],
               partial(Bottleneck, expansion=4), expansion=4)


def drn_d_105(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 23, 3, 1, 1],
               partial(Bottleneck, expansion=4), expansion=4)


def drn_d_107(in_channels, num_classes):
    return DRN(in_channels, num_classes, 'D', [3, 4, 23, 3, 2, 2],
               partial(Bottleneck, expansion=4), expansion=4)


class DRN(nn.Sequential):

    def __init__(self, in_channels, num_classes, arch,
                 block_depth, block, expansion=1):

        features = OrderedDict()

        if arch == 'A':
            features['head'] = nn.Sequential(
                ConvBlock(in_channels, 64, kernel_size=7, padding=3, stride=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif arch == 'B' or arch == 'C':
            features['head'] = nn.Sequential(
                ConvBlock(in_channels, 16, kernel_size=7, padding=3, stride=1),
                BasicBlock(16, 16),
                BasicBlock(16, 32, stride=2),
            )
        elif arch == 'D':
            features['head'] = nn.Sequential(
                ConvBlock(in_channels, 16, kernel_size=7, padding=3, stride=1),
                ConvBlock(16, 16, 3, padding=1, stride=1),
                ConvBlock(16, 32, 3, padding=1, stride=2),
            )

        features['layer1'] = make_layer(
            32, 64 * expansion, block_depth[0], block, stride=2)
        features['layer2'] = make_layer(
            64 * expansion, 128 * expansion,
            block_depth[1], block, stride=2)
        features['layer3'] = make_layer(
            128 * expansion, 256 * expansion,
            block_depth[2], partial(block, dilation=2))
        features['layer4'] = make_layer(
            256 * expansion, 512 * expansion,
            block_depth[3], partial(block, dilation=4))

        if arch == 'C':
            features['layer5'] = make_layer(
                512 * expansion, 512,
                num_blocks=block_depth[4],
                block=partial(BasicBlock, dilation=2, use_residual=False))
            features['layer6'] = make_layer(
                512, 512,
                num_blocks=block_depth[5],
                block=partial(BasicBlock, dilation=1, use_residual=False))
        elif arch == 'B':
            features['layer5'] = make_layer(
                512 * expansion, 512,
                num_blocks=block_depth[4],
                block=partial(block, dilation=2, use_residual=True))
            features['layer6'] = make_layer(
                512, 512,
                num_blocks=block_depth[5],
                block=partial(block, dilation=1, use_residual=True))
        elif arch == 'D':
            features['layer5'] = make_layer(
                512 * expansion, 512,
                num_blocks=block_depth[4],
                block=partial(ConvBlock, kernel_size=3, padding=2, dilation=2))
            features['layer6'] = make_layer(
                512, 512,
                num_blocks=block_depth[5],
                block=partial(ConvBlock, kernel_size=3, padding=1, dilation=1))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', nn.Sequential(features)),
            ('classifier', classifier),
        ]))


def make_layer(in_channels, out_channels, num_blocks, block, stride=1):
    layers = [block(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers += [block(out_channels, out_channels)]
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1,
                 use_residual=True):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels, out_channels, 3,
            padding=dilation, stride=stride, dilation=dilation)
        self.conv2 = ConvBlock(
            out_channels, out_channels, 3,
            padding=dilation, dilation=dilation, use_relu=False)

        self.use_residual = use_residual

        self.downsample = (
            None if (not use_residual
                     or (stride == 1 and in_channels == out_channels))
            else
            ConvBlock(in_channels, out_channels, 1,
                      stride=stride, use_relu=False)
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        if self.downsample is not None:
            input = self.downsample(input)
        if self.use_residual:
            return F.relu(input + x)
        else:
            return F.relu(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1,
                 expansion=4,
                 use_residual=True):
        super().__init__()

        width = out_channels // expansion
        self.conv1 = ConvBlock(in_channels, width, 1)
        self.conv2 = ConvBlock(
            width, width, 3,
            padding=dilation, stride=stride, dilation=dilation)
        self.conv3 = ConvBlock(width, out_channels, 1, use_relu=False)

        self.use_residual = use_residual

        self.downsample = (
            None if (not use_residual
                     or (stride == 1 and in_channels == out_channels))
            else
            ConvBlock(in_channels, out_channels, 1,
                      stride=stride, use_relu=False))

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            input = self.downsample(input)
        if self.use_residual:
            return F.relu(input + x)
        else:
            return F.relu(x)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, dilation=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
