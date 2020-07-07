from functools import partial
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from neural.utils.hub import configure_model

__all__ = [
    'DeepLabV2',
    'deeplabv2_resnet18', 'deeplabv2_resnet34', 'deeplabv2_resnet50', 'deeplabv2_resnet101',
]


@configure_model({
    'cocostuff': {
        'in_channels': 3,
        'out_channels': 182,
        'state_dict': 'http://files.deeplar.tk/neural/weights/deeplabv2/deeplabv2_resnet18-cocostuff-1c984de7.pth',
    },
    'bdd100k': {
        'in_channels': 3,
        'out_channels': 19,
        'state_dict': 'http://files.deeplar.tk/neural/weights/deeplabv2/deeplabv2_resnet18-bdd100k-bda23622.pth',
    }
})
def deeplabv2_resnet18(in_channels, out_channels):
    from neural.models.classification.resnet import resnet18, ResNet
    model = resnet18(in_channels, 1)
    ResNet.replace_stride_with_dilation(model, output_stride=8)

    return DeepLabV2(model.features, 512, out_channels, atrous_rates=[6, 12, 18, 24])


def deeplabv2_resnet34(in_channels, out_channels):
    from neural.models.classification.resnet import resnet34, ResNet
    model = resnet34(in_channels, 1)
    ResNet.replace_stride_with_dilation(model, output_stride=8)

    return DeepLabV2(model.features, 512, out_channels, atrous_rates=[6, 12, 18, 24])


def deeplabv2_resnet50(in_channels, out_channels):
    from neural.models.classification.resnet import resnet50, ResNet
    model = resnet50(in_channels, 1)
    ResNet.replace_stride_with_dilation(model, output_stride=8)

    return DeepLabV2(model.features, 2048, out_channels, atrous_rates=[6, 12, 18, 24])


def deeplabv2_resnet101(in_channels, out_channels):
    from neural.models.classification.resnet import resnet101, ResNet
    model = resnet101(in_channels, 1)
    ResNet.replace_stride_with_dilation(model, output_stride=8)

    return DeepLabV2(model.features, 2048, out_channels, atrous_rates=[6, 12, 18, 24])


class DeepLabV2(nn.Sequential):
    def __init__(self, feature_extractor, feature_channels, out_channels,
                 atrous_rates=[6, 12, 18, 24]):
        super(DeepLabV2, self).__init__()
        self.features = feature_extractor
        self.aspp = ASPPModule(feature_channels, feature_channels, atrous_rates)
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(feature_channels, out_channels, 1),
        )

    def forward(self, input):
        x = self.features(input)
        x = self.aspp(x)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:],
                             mode='bilinear', align_corners=True)


class ASPPModule(nn.ModuleList):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__([
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=True)
            for rate in atrous_rates
        ])

    def forward(self, input):
        return sum((pool(input) for pool in self.children()))


class MultiscaleContext(nn.Module):
    def __init__(self, module, scales=[0.5, 0.75, 1]):
        self.module = module
        self.scales = scales
        self.interpolate = partial(F.interpolate, mode='bilinear', align_corners=True)

    def forward(self, input):
        inputs = (self.interpolate(input, scale_factor=scale) for scale in self.scales)
        outputs = (self.interpolate(self.module(input), size=input.shape[2:]) for input in inputs)
        return max(outputs)
