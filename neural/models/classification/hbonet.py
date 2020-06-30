from collections import OrderedDict
from functools import partial
from torch import nn


class HBONet(nn.Sequential):
    def __init__(self, in_channels, num_classes, width_multiplier=1, spatial_rate=4):

        def c(channels):
            divisor = 4 if width_multiplier >= 0.5 else 2
            return round_by(channels, divisor)

        Block = partial(HarmoniousBlock, expansion_rate=1, spatial_rate=spatial_rate)

        def make_block(in_channels, out_channels, num_blocks=1, stride=1, block=Block, **kwargs):
            layers = [block(in_channels, out_channels, stride=stride, **kwargs)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels, **kwargs)]
            return nn.Sequential(*layers)

        out_channels = max(1600, c(1600))

        features = nn.Sequential(
            ConvBlock(in_channels, c(32), 3, padding=1, stride=2),
            make_block(c(32), c(20), expansion=1),
            make_block(c(20), c(36)),
            make_block(c(36), c(72), 3, stride=2),
            make_block(c(72), c(96), 4, stride=2),
            make_block(c(96), c(192), 4, stride=2),
            make_block(c(192), c(288)),
            ConvBlock(c(288), c(144), 1),
            make_block(c(144), c(200), 1, stride=2, block=InvertedResidualBlock),
            make_block(c(200), c(400), block=InvertedResidualBlock),
            ConvBlock(c(400), out_channels),
        )

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes),
        )

        super(HBONet, self).__init__()


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(InvertedResidualBlock, self).__init__()

        hidden_channels = round_by(in_channels * expansion)
        self.conv1 = (
            ConvBlock(in_channels, hidden_channels, 1)
            if expansion != 1 else nn.Identity())
        self.conv2 = DWConvBlock(hidden_channels, hidden_channels, 3, padding=1, stride=stride)
        self.conv3 = ConvBlock(hidden_channels, out_channels, 1, use_relu=False)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        return x + input if input.shape == x.shape else x


class HarmoniousBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, spatial_rate=0):
        super(HarmoniousBlock, self).__init__()

        num_blocks = {2: 0, 4: 1, 8: 2}[spatial_rate]

        hidden_channels = round_by(in_channels * expansion)
        self.conv1 = nn.Sequential(
            DWConvBlock(in_channels, in_channels, 5, padding=2, stride=2, use_relu=False),
            ConvBlock(in_channels, hidden_channels),
        )
        self.conv2 = nn.Sequential(*[
            nn.Sequential(
                DWConvBlock(hidden_channels, hidden_channels, 5, padding=2, stride=2, use_relu=False),
                ConvBlock(hidden_channels, hidden_channels, 1),
            )
            for _ in range(num_blocks)
        ])
        self.conv3 = nn.Sequential(
            DWConvBlock(hidden_channels, hidden_channels, 3, padding=1, use_relu=False),
            ConvBlock(hidden_channels, out_channels // 2),
        )

        scale_factor = spatial_rate if stride == 1 else spatial_rate // 2
        self.upsample = nn.Upsample(scale_factor=scale_factor) if scale_factor > 1 else nn.Identity()
        self.pool = nn.AvgPool2D(2) if stride == 2 else nn.Identity()

        self.upconv = DWConvBlock(out_channels // 2, out_channels // 2, 5, padding=2, stride=2, use_relu=False)

    def forward(self, input):
        left, right = torch.chunk(input, 2, dim=1)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        return torch.cat([
            self.pool(right),
            self.upconv(self.pool(left) + self.upsample(x))
        ], dim=1)


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU6(inplace=True)]
    return nn.Sequential(*layers)


def DWConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, use_relu=True):
    return ConvBlock(in_channels, out_channels, kernel_size,
                     padding=padding, stride=stride,
                     groups=in_channels, use_relu=True)


def round_by(channels, divisor=8):
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
