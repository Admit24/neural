from collections import OrderedDict
from functools import partial
from torch import nn
from torch.hub import load_state_dict_from_url

__all__ = [
    'Vgg',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
]

model_urls = {
    'vgg11': {
        'imagenet': 'http://files.deeplar.tk/neural/models/weights/classification/vgg/vgg11-imagenet-c4ec67ec.pth', },
    'vgg13': {
        'imagenet': 'http://files.deeplar.tk/neural/models/weights/classification/vgg/vgg13-imagenet-aa4ef656.pth', },
    'vgg16': {
        'imagenet': 'http://files.deeplar.tk/neural/models/weights/classification/vgg/vgg16-imagenet-34571d8c.pth', },
    'vgg19': {
        'imagenet': 'http://files.deeplar.tk/neural/models/weights/classification/vgg/vgg19-imagenet-ac55300b.pth', },
}


def vgg(model_name, in_channels=3, num_classes=1000, pretrained=None):
    Model = {'vgg11': Vgg11, 'vgg13': Vgg13,
             'vgg16': Vgg16, 'vgg19': Vgg19}[model_name]

    if pretrained:
        if pretrained == 'imagenet':
            model = Model(in_channels=3, num_classes=1000)
        else:
            raise ValueError('dataset not found')

        state_dict = load_state_dict_from_url(
            model_urls[model_name][pretrained], check_hash=True)
        model.load_state_dict(state_dict)
        return model

    else:
        return Model(in_channels, num_classes)


vgg11 = partial(vgg, 'vgg11')
vgg13 = partial(vgg, 'vgg13')
vgg16 = partial(vgg, 'vgg16')
vgg19 = partial(vgg, 'vgg19')


class Vgg(nn.Sequential):

    def __init__(self, in_channels, num_classes, block_depth):

        features = nn.Sequential(OrderedDict([
            ('layer1', make_layer(in_channels, 64, block_depth[0])),
            ('layer2', make_layer(64, 128, block_depth[1])),
            ('layer3', make_layer(128, 256, block_depth[2])),
            ('layer4', make_layer(256, 512, block_depth[3])),
            ('layer5', make_layer(512, 512, block_depth[4])),
        ]))

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


Vgg11 = partial(Vgg, block_depth=[1, 1, 2, 2, 2])
Vgg13 = partial(Vgg, block_depth=[2, 2, 2, 2, 2])
Vgg16 = partial(Vgg, block_depth=[2, 2, 3, 3, 3])
Vgg19 = partial(Vgg, block_depth=[2, 2, 4, 4, 4])


def make_layer(in_channels, out_channels, num_blocks):
    layers = []
    layers.append(ConvBlock(in_channels, out_channels))
    for _ in range(1, num_blocks):
        layers.append(ConvBlock(out_channels, out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
