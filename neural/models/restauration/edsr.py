from torch import nn
from torch.nn import functional as F

from neural.utils.hub import configure_model

__all__ = [
    'EDSR',
    'edsr', 'edsr_baseline',
]


@configure_model({
    'div2k_2x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 2,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_2x-div2k-b109c297.pth',
    },
    'div2k_3x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 3,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_3x-div2k-39a5f2e2.pth',
    },
    'div2k_4x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 4,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_4x-div2k-586f4bfa.pth',
    },
})
def edsr(in_channels, out_channels, scale_factor):
    return EDSR(in_channels, out_channels, scale_factor,
                width=256, num_blocks=32, resblock_scaling=0.1)


@configure_model({
    'div2k_2x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 2,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_baseline_2x-div2k-f821ffdc.pth',
    },
    'div2k_3x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 3,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_baseline_3x-div2k-c7c2097c.pth',
    },
    'div2k_4x': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factor': 4,
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/edsr_baseline_4x-div2k-275cd61d.pth',
    },
})
def edsr_baseline(in_channels, out_channels, scale_factor):
    return EDSR(in_channels, out_channels, scale_factor,
                width=64, num_blocks=16, resblock_scaling=None)


@configure_model({
    'div2k': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factors': [2, 3, 4],
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/mdsr-div2k-50920243.pth',
    }
})
def mdsr(in_channels, out_channels, scale_factors=[2, 3, 4]):
    return MDSR(in_channels, out_channels, scale_factors=scale_factors, width=64, num_blocks=80)


@configure_model({
    'div2k': {
        'in_channels': 3,
        'out_channels': 3,
        'scale_factors': [2, 3, 4],
        'state_dict': 'http://files.deeplar.tk/neural/weights/edsr/mdsr_baseline-div2k-c4070d35.pth',
    }
})
def mdsr_baseline(in_channels, out_channels, scale_factors=[2, 3, 4]):
    return MDSR(in_channels, out_channels, scale_factors=scale_factors, width=64, num_blocks=16)


class EDSR(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 width=64,
                 num_blocks=16,
                 resblock_scaling=None):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels, width,
            kernel_size=3, padding=1)

        layers = [
            ResidualBlock(width, width, scaling=resblock_scaling)
            for _ in range(num_blocks)
        ] + [nn.Conv2d(width, width, 3, padding=1)]
        self.features = nn.Sequential(*layers)

        self.upsampling = create_upsampling(width, scale_factor)

        self.tail = nn.Conv2d(
            width, out_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        features = self.features(x)
        x = x + features
        x = self.upsampling(x)
        return self.tail(x)


class MDSR(nn.Module):
    def __init__(self, in_channels, out_channels,
                 scale_factors=[2, 3, 4],
                 width=64,
                 num_blocks=16,
                 resblock_scaling=None):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, width, 3, padding=1)

        self.head = nn.ModuleDict({
            str(scale_factor): nn.Sequential(
                ResidualBlock(width, width, 5),
                ResidualBlock(width, width, 5),
            )
            for scale_factor in scale_factors
        })

        layers = [
            ResidualBlock(width, width, scaling=resblock_scaling)
            for _ in range(num_blocks)
        ] + [nn.Conv2d(width, width, 3, padding=1)]
        self.features = nn.Sequential(*layers)

        self.upsampling = nn.ModuleDict({
            str(scale_factor): create_upsampling(width, scale_factor)
            for scale_factor in scale_factors
        })

        self.tail = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, input, scale_factor):
        x = self.conv(input)
        x = self.head[str(scale_factor)](x)
        features = self.features(x)
        x = x + features
        x = self.upsampling[str(scale_factor)](x)
        return self.tail(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, scaling=1.0):
        super().__init__()

        self.scaling = scaling

        self.conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.conv2(x)
        if self.scaling is not None:
            x = self.scaling * x
        return input + x


def create_upsampling(width, scale_factor):
    if scale_factor in {2, 3}:
        return Upsampling(width, scale_factor)
    elif scale_factor == 4:
        return nn.Sequential(
            Upsampling(width, 2),
            Upsampling(width, 2),
        )
    else:
        raise ValueError(
            "scale_factor should be either 2, 3 or 4, "
            "got {}".format(scale_factor))


class Upsampling(nn.Module):

    def __init__(self, channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            channels, channels * scale_factor**2,
            kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(
            upscale_factor=scale_factor)

    def forward(self, input):
        return self.shuffle(self.conv(input))
