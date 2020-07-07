from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from neural.utils.hub import configure_model


@configure_model({
    'cityscapes': {
        'in_channels': 3, 'out_channels': 19,
        'state_dict': 'http://files.deeplar.tk/neural/weights/segnet/segnet-cityscapes-ae73a541.pth',
    }
})
def segnet(in_channels=3, out_channels=19):
    return SegNet(in_channels, out_channels)


class SegNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = nn.ModuleDict(OrderedDict([
            ('layer1', make_layer(in_channels, 64, 2)),
            ('layer2', make_layer(64, 128, 2)),
            ('layer3', make_layer(128, 256, 3)),
            ('layer4', make_layer(256, 512, 3)),
            ('layer5', make_layer(512, 512, 3)),
        ]))

        self.decoder = nn.ModuleDict(OrderedDict([
            ('layer1', make_layer(512, 512, 3)),
            ('layer2', make_layer(512, 256, 3)),
            ('layer3', make_layer(256, 128, 3)),
            ('layer4', make_layer(128, 64, 2)),
            ('layer5', make_layer(64, out_channels, 2)),
        ]))

    def forward(self, input):
        pooling_indices = []
        x = input
        for layer in self.encoder.children():
            x = layer(x)
            x, indices = F.max_pool2d_with_indices(x, kernel_size=2)
            pooling_indices = [indices, *pooling_indices]
        for layer in self.decoder.children():
            indices, *pooling_indices = pooling_indices
            x = F.max_unpool2d(x, indices, kernel_size=2)
            x = layer(x)
        return x


def make_layer(in_channels, out_channels, num_blocks):
    layers = [ConvBlock(in_channels, out_channels)]
    for _ in range(1, num_blocks):
        layers += [ConvBlock(out_channels, out_channels)]
    return nn.Sequential(*layers)


def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
