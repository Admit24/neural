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
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


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
                num_channels=64, num_res_blocks=16, resblock_scaling=None)


class EDSR(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 num_channels=64,
                 num_res_blocks=16,
                 resblock_scaling=None):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels, num_channels,
            kernel_size=3, padding=1)
        self.features = nn.Sequential(*[
            ResidualBlock(num_channels, num_channels, scaling=resblock_scaling)
            for _ in range(num_res_blocks)
        ] + [nn.Conv2d(num_channels, num_channels, 3, padding=1)])
        if scale_factor in {2, 3}:
            self.upsampling = Upsampling(num_channels, scale_factor)
        elif scale_factor == 4:
            self.upsampling = nn.Sequential(
                Upsampling(num_channels, 2),
                Upsampling(num_channels, 2),
            )
        else:
            raise ValueError(
                "scale_factor should be either 2, 3 or 4, "
                "got {}".format(scale_factor))
        self.tail = nn.Conv2d(
            num_channels, out_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        features = self.features(x)
        x = x + features
        x = self.upsampling(x)
        return self.tail(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scaling=1.0):
        super().__init__()

        self.scaling = scaling

        self.conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.conv2(x)
        if self.scaling is not None:
            x = self.scaling * x
        return input + x


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
