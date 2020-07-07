from collections import OrderedDict

from torch import nn
from torch.nn import functional as F
from neural.utils.hub import configure_model

__all__ = [
    'ResNet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
]


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/resnet/resnet18-imagenet-ba211383.pth',
    }
})
def resnet18(in_channels=3, num_classes=1000):
    return ResNet(in_channels, num_classes,
                  block_depth=[2, 2, 2, 2],
                  block=BasicBlock)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/resnet/resnet34-imagenet-c9f799d0.pth',
    }
})
def resnet34(in_channels=3, num_classes=1000):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=BasicBlock)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/resnet/resnet50-imagenet-eb6991ee.pth',
    }
})
def resnet50(in_channels=3, num_classes=1000):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=BottleneckBlock,
                  expansion=4)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/resnet/resnet101-imagenet-b8d0e605.pth',
    }
})
def resnet101(in_channels=3, num_classes=1000):
    return ResNet(in_channels, num_classes,
                  block_depth=[2, 3, 23, 3],
                  block=BottleneckBlock,
                  expansion=4)


@configure_model({
    'imagenet': {
        'state_dict': 'http://files.deeplar.tk/neural/weights/resnet/resnet152-imagenet-e3a13154.pth',
    }
})
def resnet152(in_channels=3, num_classes=1000):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 8, 36, 3],
                  block=BottleneckBlock,
                  expansion=4)


class ResNet(nn.Sequential):

    def __init__(self, in_channels, num_classes,
                 block_depth, block, expansion=1):

        def make_layer(in_channels, out_channels, num_blocks, block, stride=1):
            layers = [block(in_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )),
            ('layer1', make_layer(
                64, 64 * expansion, block_depth[0], block=block, stride=1)),
            ('layer2', make_layer(
                64 * expansion, 128 * expansion, block_depth[1],
                block=block, stride=2)),
            ('layer3', make_layer(
                128 * expansion, 256 * expansion, block_depth[2],
                block=block, stride=2)),
            ('layer4', make_layer(
                256 * expansion, 512 * expansion, block_depth[3],
                block=block, stride=2)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * expansion, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))

    @staticmethod
    def replace_stride_with_dilation(model, output_stride, multigrid_rates=None):
        if output_stride not in [8, 16, 32]:
            raise ValueError("output stride should be {8, 16, 32}. Got {}."
                             .format(output_stride))

        def patch_layer(layer, dilation, multigrid_rates=None):
            # change the stride of the first block
            if isinstance(layer[0], BasicBlock):
                layer[0].conv1.stride = 1
            elif isinstance(layer[0], BottleneckBlock):
                layer[0].conv2.stride = 1
            layer[0].downsample[0].stride = 1
            # change the dilation rate of all the convolutions in the layer
            for id, m in enumerate(layer.children()):
                rate = 1 if multigrid_rates is None else multigrid_rates[id]

                if isinstance(m, BasicBlock):
                    m.conv1.dilation = rate * dilation
                    m.conv1.padding = rate * dilation
                if isinstance(m, BottleneckBlock):
                    m.conv2.dilation = rate * dilation
                    m.conv2.padding = rate * dilation

        if output_stride == 8:
            patch_layer(model.features.layer3, dilation=2)
            patch_layer(model.features.layer4, dilation=4, multigrid_rates=multigrid_rates)
        elif output_stride == 16:
            patch_layer(model.features.layer4, dilation=2, multigrid_rates=multigrid_rates)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=3,
            padding=1, stride=stride)
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3,
            padding=1, use_relu=False)

        self.downsample = (
            ConvBlock(
                in_channels, out_channels, kernel_size=1,
                stride=stride, use_relu=False)
            if in_channels != out_channels or stride == 2 else None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        s = (input
             if self.downsample is None
             else self.downsample(input))
        return F.relu(x + s)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()

        width = out_channels // expansion

        self.conv1 = ConvBlock(
            in_channels, width, kernel_size=1)
        self.conv2 = ConvBlock(
            width, width, kernel_size=3,
            padding=1, stride=stride)
        self.conv3 = ConvBlock(
            width, out_channels,
            kernel_size=1, use_relu=False)

        self.downsample = (
            ConvBlock(
                in_channels, out_channels, kernel_size=1,
                stride=stride, use_relu=False)
            if in_channels != out_channels or stride == 2 else None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        s = (input
             if self.downsample is None
             else self.downsample(input))
        return F.relu(x + s)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1,
              use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
