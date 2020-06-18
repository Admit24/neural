
# classification models
from neural.models.classification.alexnet import alexnet
from neural.models.classification.vgg import vgg11, vgg13, vgg16, vgg19

# segmentation models
from neural.models.segmentation.segnet import segnet
from neural.models.segmentation.fasterscnn import fastscnn
from neural.models.segmentation.enet import enet
from neural.models.segmentation.contextnet import (
    contextnet12, contextnet14, contextnet18)

dependencies = ['torch']
