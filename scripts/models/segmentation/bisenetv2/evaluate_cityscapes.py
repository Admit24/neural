import argparse
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from ignite.contrib.handlers import ProgressBar

from neural.models.segmentation.bisenetv2 import bisenetv2
from neural.engines.segmentation import create_segmentation_evaluator
from neural.data.cityscapes import Cityscapes

from neural.utils.training import get_datasets_root
from neural.data.cityscapes import CLASSES, TRAIN_MAPPING

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--state_dict', type=str, required=True)
args = parser.parse_args()


# Define device
device = torch.device('cuda')

# Define transformations to Validation set
val_tfms = albu.Compose([
    albu.Normalize(
        mean=(0.3257, 0.3690, 0.3223),
        std=(0.2112, 0.2148, 0.2115),
    ),
    ToTensor(),
])
# Get Validation set of cityscapes
dataset_dir = get_datasets_root('cityscapes')
val_dataset = Cityscapes(dataset_dir, split='val', transforms=val_tfms)


# Load dataset
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)
# Create model

model = bisenetv2(3, 19)
model.eval()

# Load model
state_dict = torch.load(args.state_dict, map_location='cpu')
model.load_state_dict(state_dict, strict=True)

# model = torch.jit.load(args.state_dict)
model = model.to(device)


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)[0]


evaluator = create_segmentation_evaluator(
    model,
    device=device,
    num_classes=19,
)

ProgressBar().attach(evaluator)

state = evaluator.run(val_loader)

classes = CLASSES[TRAIN_MAPPING != 255]


metrics = {
    'accuracy': state.metrics['accuracy'],
    'miou': state.metrics['miou'],
    'iou': {name: state.metrics['iou'][id].item() for id, name in enumerate(classes)},
}

pprint(metrics)
