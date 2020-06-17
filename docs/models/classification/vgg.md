# VGG

Based on the work of __"Very Deep Convolutional Networks for Large-Scale Image Recognition"__, by Karen Simonyan and Andrew Zisserman [Arxiv](https://arxiv.org/abs/1409.1556).

## Description

VGG is one of most useful and used models made for the large-scale classification task of ImageNet, due to it's well defined and concise structure. This model is composed of stacking layers of 3x3 convolution and pooling operations. 

## Use Guide

```py
import torch

# if you want to use this repository as the entry point
from neural.models.classification.vgg import vgg16

model = vgg16(in_channels=3, num_classes=1000)

# if you want to use the torch hub to load the model
model = torch.hub.load('bernardomig/neural', 'vgg16')

# if you want to use the compiled model
model = torch.jit.load('<downloaded model>')

# apply the transformations to the image
from torchvision import transforms
from PIL import Image

image = Image.open('my-image.jpg')

tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])
x = tfms(image)
batch = x.unsqueeze(0)

# use the gpu if possible
if torch.cuda.is_available:
    batch = batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(x)

# print the id of the predicted class
print('Prediction: {}'.format(torch.argmax(output[0])))
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
