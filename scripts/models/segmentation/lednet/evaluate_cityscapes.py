import argparse
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from ignite.contrib.handlers import ProgressBar

from neural.models.segmentation.lednet import lednet
from neural.engines.segmentation import create_segmentation_evaluator
from neural.data.cityscapes import Cityscapes

from neural.utils.training import get_datasets_root
from neural.data.cityscapes import CLASSES, TRAIN_MAPPING

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--state_dict', type=str, required=True)
args = parser.parse_args()


# Define device
device = torch.device('cpu')

# Define transformations to Validation set
val_tfms = albu.Compose([
    albu.Normalize(),
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
model = lednet(3, 19)

# Load model
state_dict = torch.load(args.state_dict, map_location='cpu')
model.load_state_dict(state_dict, strict=True)

model = model.to(device)

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
