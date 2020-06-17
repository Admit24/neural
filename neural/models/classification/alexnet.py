from collections import OrderedDict
from torch import nn

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': {
        'imagenet': 'http://files.deeplar.tk/neural/weights/classification/alexnet/alexnet-imagenet-2db60a67.pth'
    },
}


def alexnet(in_channels=3, num_classes=1000, pretrained=None):
    if pretrained:
        from torch.hub import load_state_dict_from_url
        if pretrained == 'imagenet':
            model = AlexNet(in_channels=3, num_classes=1000)
            state_dict = load_state_dict_from_url(
                model_urls['alexnet']['imagenet'],
                check_hash=True)
            model.load_state_dict(state_dict)
            return model
        else:
            raise ValueError('dataset not found')
    else:
        return AlexNet(in_channels=in_channels, num_classes=1000)


class AlexNet(nn.Sequential):

    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, 64,
                                   kernel_size=11, stride=4, padding=2)),
                ('lrn', nn.LocalResponseNorm(64, k=2)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
            ('layer2', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('lrn', nn.LocalResponseNorm(192, k=2)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
            ('layer3', nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
        ]))

        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))
