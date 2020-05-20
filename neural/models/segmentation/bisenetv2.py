from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'BisenetV2',
    'bisenetv2',
    'bisenetv2_large',
]


def bisenetv2(in_channels, out_channels):
    return BisenetV2(in_channels, out_channels,
                     width_multiplier=1, depth_multiplier=1)


def bisenetv2_large(in_channels, out_channels):
    return BisenetV2(in_channels, out_channels,
                     width_multiplier=2, depth_multiplier=3)


class BisenetV2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 width_multiplier=1, depth_multiplier=1):
        super(BisenetV2, self).__init__()

        def c(channels): return int(width_multiplier * channels)
        def d(depth): return int(depth_multiplier * depth)

        def make_conv_layer(in_channels, out_channels, repeats, stride=1):
            layers = [ConvBlock(in_channels, out_channels, 3,
                                padding=1, stride=stride)]
            for _ in range(1, repeats):
                layers += [
                    ConvBlock(out_channels, out_channels, 3,
                              padding=1)
                ]
            return nn.Sequential(*layers)

        self.detail = nn.Sequential(OrderedDict([
            ('stage1', make_conv_layer(in_channels, c(64), d(2), stride=2)),
            ('stage2', make_conv_layer(c(64), c(64), d(3), stride=2)),
            ('stage3', make_conv_layer(c(64), c(128), d(3), stride=2)),
        ]))

        def make_block(in_channels, out_channels, repeats, stride=1):
            layers = [GatherExpansionBlock(
                in_channels, out_channels,
                stride=stride)]
            for _ in range(1, repeats):
                layers += [
                    GatherExpansionBlock(out_channels, out_channels)
                ]
            return nn.Sequential(*layers)

        self.semantic = nn.Sequential(OrderedDict([
            ('stem', StemBlock(in_channels, c(16))),
            ('stage3', make_block(c(16), c(32), d(2), 2)),
            ('stage4', make_block(c(32), c(64), d(2), 2)),
            ('stage5', nn.Sequential(
                *make_block(c(64), c(128), d(4), stride=2),
                ContextEmbeddingBlock(c(128), c(128)),
            ))
        ]))

        self.aggregation = BilateralGuidedAggregationBlock(c(128), c(128))

        self.classifier = nn.Sequential(
            ConvBlock(c(128), c(128), 3, padding=1),
            nn.Conv2d(c(128), out_channels, 1),
        )

    def forward(self, input):
        detail = self.detail(input)
        semantic = self.semantic(input)
        x = self.aggregation(detail, semantic)
        x = self.classifier(x)
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)


class StemBlock(nn.Module):
    # The stem block adopts a fast downsampling strategy.
    # This block has two branches with different manners to downsample the
    # feature representation. Then both feature response from both branches
    # are concatenated as the output.
    # This structure has efficient computationcost and effective feature
    # expression ability.

    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, 3,
                               padding=1, stride=2)
        self.conv2 = nn.Sequential(
            ConvBlock(out_channels, out_channels // 2, 1),
            ConvBlock(out_channels // 2, out_channels, 3, padding=1, stride=2),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.conv3 = ConvBlock(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = torch.cat([
            self.conv2(x),
            self.pool2(x)
        ], dim=1)
        return self.conv3(x)


class BilateralGuidedAggregationBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BilateralGuidedAggregationBlock, self).__init__()

        self.detail = nn.ModuleDict({
            'conv': nn.Sequential(
                DWConvBlock(in_channels, in_channels, 3,
                            padding=1, use_relu=False),
                nn.Conv2d(in_channels, in_channels, 1),
            ),
            'pool': nn.Sequential(
                ConvBlock(in_channels, in_channels, 3,
                          padding=1, stride=2, use_relu=False),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            ),
        })

        self.semantic = nn.ModuleDict({
            'conv': nn.Sequential(
                DWConvBlock(in_channels, in_channels, 3,
                            padding=1, use_relu=False),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.Sigmoid(),
            ),
            'pool': nn.Sequential(
                ConvBlock(in_channels, in_channels, 3,
                          padding=1, use_relu=False),
                nn.Upsample(scale_factor=4,
                            mode='bilinear', align_corners=True),
                nn.Sigmoid(),
            ),
        })

        self.conv = ConvBlock(in_channels, out_channels, 3,
                              padding=1, use_relu=False)

    def forward(self, detail, semantic):
        left = torch.mul(
            self.detail['conv'](detail),
            self.semantic['pool'](semantic),
        )
        right = torch.mul(
            self.detail['pool'](detail),
            self.semantic['conv'](semantic),
        )

        x = torch.add(
            left,
            F.interpolate(right, scale_factor=4,
                          mode='bilinear', align_corners=True),
        )

        return self.conv(x)


class GatherExpansionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(GatherExpansionBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels, 3, padding=1)
        self.conv2 = (
            DWConvBlock(
                in_channels, 6 * in_channels, 3,
                padding=1, use_relu=False)
            if stride == 1 else
            nn.Sequential(
                DWConvBlock(in_channels, 6 * in_channels, 3,
                            padding=1, stride=stride, use_relu=False),
                DWConvBlock(6 * in_channels, 6 * in_channels, 3,
                            padding=1, use_relu=False),
            )
        )
        self.conv3 = ConvBlock(
            6 * in_channels, out_channels, 1,
            use_relu=False)

        self.downsample = (
            None
            if stride == 1 else
            nn.Sequential(
                DWConvBlock(in_channels, in_channels, 3,
                            padding=1, stride=stride, use_relu=False),
                ConvBlock(in_channels, out_channels, 1, use_relu=False)
            )
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            input = self.downsample(input)

        return x + input


class ContextEmbeddingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ContextEmbeddingBlock, self).__init__()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.BatchNorm2d(in_channels),
            ConvBlock(in_channels, in_channels, 1),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, input):
        x = self.pool(input)
        x = x + input
        return self.conv(x)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1,
              groups=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=groups,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def DWConvBlock(in_channels, out_channels, kernel_size,
                padding=0, stride=1,
                use_relu=True):
    return ConvBlock(in_channels, out_channels, kernel_size,
                     padding=padding, stride=stride,
                     use_relu=use_relu,
                     groups=in_channels)
