from torch import nn
from torch.nn import functional as F


def icnet_resnet18(in_channels, out_channels):
    from neural.models.classification.resnet import ResNet, resnet18
    backbone = resnet18(in_channels, 1)
    ResNet.replace_stride_with_dilation(backbone, output_stride=8)
    features = backbone.features
    features.add_module('spp', PyramidPoolingModule())
    return ICNet(
        ICNet_Encoder(in_channels, nn.ModuleList([features[:3], features[3:]])),
        ICNet_Head((64, 128, 512), 128),
        nn.Conv2d(128, out_channels, 1),
    )


def icnet_resnet34(in_channels, out_channels):
    from neural.models.classification.resnet import ResNet, resnet34
    backbone = resnet34(in_channels, 1)
    ResNet.replace_stride_with_dilation(backbone, output_stride=8)
    features = backbone.features
    features.add_module('spp', PyramidPoolingModule())
    return ICNet(
        ICNet_Encoder(in_channels, nn.ModuleList([features[:3], features[3:]])),
        ICNet_Head((64, 128, 512), 128),
        nn.Conv2d(128, out_channels, 1),
    )


def icnet_resnet50(in_channels, out_channels):
    from neural.models.classification.resnet import ResNet, resnet50
    backbone = resnet50(config='imagenet')
    ResNet.replace_stride_with_dilation(backbone, output_stride=8)
    features = backbone.features
    features.add_module('spp', PyramidPoolingModule())
    return ICNet(
        ICNet_Encoder(in_channels, nn.ModuleList([features[:3], features[3:]])),
        ICNet_Head((64, 512, 2048), 128),
        nn.Conv2d(128, out_channels, 1),
    )


class ICNet(nn.Module):
    def __init__(self, encoder, head, classifier):
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.classifier = classifier

    def forward(self, input):
        x = self.encoder(input)
        x = self.head(x)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)


class ICNet_Encoder(nn.Module):

    def __init__(self, in_channels, backbone):
        super().__init__()

        self.spatial = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, padding=1, stride=2),
            ConvBNReLU(32, 32, 3, padding=1, stride=2),
            ConvBNReLU(32, 64, 3, padding=1, stride=2),
        )

        self.context = backbone

    def forward(self, input):
        x2 = F.avg_pool2d(input, 2)
        x4 = F.avg_pool2d(x2, 2)

        return (
            self.spatial(input),
            self.context[0](x2),
            self.context[1](self.context[0](x4)),
        )


class ICNet_Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        channels1, channels2, channels4 = in_channels

        self.cff12 = CascadeFeatureFusion((channels1, out_channels), out_channels)
        self.cff24 = CascadeFeatureFusion((channels2, channels4), out_channels)

    def forward(self, input):
        x1, x2, x4 = input

        x = self.cff24(x2, x4)
        x = self.cff12(x1, x)

        return x


class CascadeFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.highres = ConvBNReLU(in_channels[0], out_channels, 3, padding=2, dilation=2, include_relu=False)
        self.lowres = ConvBNReLU(in_channels[1], out_channels, 3, 1, include_relu=False)

    def forward(self, highres, lowres):
        lowres = F.interpolate(lowres, size=highres.shape[2:], mode='bilinear', align_corners=True)
        lowres = self.lowres(lowres)
        highres = self.highres(highres)

        return F.relu(lowres + highres)


class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1, 2, 3, 6]):
        super().__init__()

        self.pyramids = pyramids

    def forward(self, input):
        features = input
        for pyramid in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=pyramid)
            x = F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)
            features = features + x
        return features


def ConvBNReLU(in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, include_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                  stride=stride, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if include_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
