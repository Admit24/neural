
# classification models
from neural.models.classification.alexnet import alexnet
from neural.models.classification.vgg import vgg11, vgg13, vgg16, vgg19
from neural.models.classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from neural.models.classification.mobilenetv2 import mobilenetv2
from neural.models.classification.ghostnet import ghostnet_1_0
from neural.models.classification.efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
    efficientnet_b5, efficientnet_b6, efficientnet_b7)
from neural.models.classification.drn import (
    drn_a_18, drn_a_34, drn_a_50, drn_c_26, drn_c_42, drn_c_58,
    drn_d_22, drn_d_24, drn_d_38, drn_d_40, drn_d_54, drn_d_56,
    drn_d_105, drn_d_107)

# segmentation models
from neural.models.segmentation.segnet import segnet
from neural.models.segmentation.fastscnn import fastscnn
from neural.models.segmentation.enet import enet
from neural.models.segmentation.contextnet import (
    contextnet12, contextnet14, contextnet18)
from neural.models.segmentation.deeplabv2 import (
    deeplabv2_resnet18, deeplabv2_resnet34, deeplabv2_resnet50, deeplabv2_resnet101)

dependencies = ['torch']
