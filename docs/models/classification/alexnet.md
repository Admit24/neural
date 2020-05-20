# AlexNet

Based on the work __"ImageNet Classification with Deep ConvolutionalNeural Networks"__, by A. Krizhevsky, I. Sutskever and G. Hinton [NIPS Proceedings](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

## Description

This was the first every convolution deep neural model to win the ImageNet classification challenge. The model is composed by a stack of convolutional layers and ReLU activations. 

###  Flavours

| Name    | # Parameters | MulA Ops |
| ------- | ------------ | -------- |
| AlexNet |              |          |

### Use Guide

```py
from neural.models.classification.alexnet import alexnet

model = alexnet(in_channels=3, num_classes=1000)
```

## Pretrained Models

| Model   | Dataset  | Top1  | Top5 error | Weights | TorchScript | ONNX |
| ------- | -------- | ----- | ---------- | ------- | ----------- | ---- |
| AlexNet | ImageNet | 56.55 | 79.09      |         |             |      |