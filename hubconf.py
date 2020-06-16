import torch

from neural.models.classification.vgg import (
    vgg11 as _vgg11,
)


dependencies = ['torch']


# The VGG models
def vgg11(dataset='imagenet', pretrained=True):
    if dataset == 'imagenet':
        model = _vgg11(in_channels=3, num_classes=1000)
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                'http://files.deeplar.tk/neural/models/weights/classification/vgg/vgg11-imagenet-c4ec67ec.pth',
                check_hash=True)
            model.load_state_dict(state_dict)
        return model
