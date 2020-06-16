# VGG

Based on the work of __"Very Deep Convolutional Networks for Large-Scale Image Recognition"__, by Karen Simonyan and Andrew Zisserman [Arxiv](https://arxiv.org/abs/1409.1556).

## Description

VGG is one of most useful and used models made for the large-scale classification task of ImageNet, due to it's well defined and concise structure. This model is composed of stacking layers of 3x3 convolution and pooling operations. 

## Models

| Name  | #Parameters | #MulA Ops |
| ----- | ----------- | --------- |
| VGG11 |             |           |
| VGG13 |             |           |
| VGG16 |             |           |
| VGG19 |             |           |

## Use Guide

```py
from neural.models.classification.vgg import vgg16

model = vgg16(in_channels=3, num_classes=1000)
```

## Pretrained Models

> **Note**: Unlike torch vision, all the models presented have batch-normalization by default after each convolution operation. We find no usefullness in not providing the batchnorm layer, and the performance is much superior.

### ImageNet (from torchvision)

| Name  | Top1-error | Top-5 error | Weights | Torch JIT | ONNX |
| ----- | ---------- | ----------- | ------- | --------- | ---- |
| VGG11 | 26.70      | 8.58        |         |           |      |
| VGG13 | 28.45      | 9.63        |         |           |      |
| VGG16 | 26.63      | 8.50        |         |           |      |
| VGG19 | 25.76      | 8.15        |         |           |      |
