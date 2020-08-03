import argparse

import torch
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from ignite.contrib.handlers import ProgressBar

from neural.engines.segmentation import create_segmentation_evaluator
from neural.data.cityscapes import Cityscapes

from neural.utils.training import get_datasets_root
from neural.data.cityscapes import CLASSES, TRAIN_MAPPING

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--state_dict', type=str, required=False)
args = parser.parse_args()


# Define device
device = torch.device('cuda')

# Define transformations to Validation set
val_tfms = albu.Compose([
    albu.Normalize(
        mean=[0.28689553, 0.32513301, 0.28389176],
        std=[0.17613641, 0.18099167, 0.17772231],
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
kwargs = (
    dict(config='cityscapes')
    if args.state_dict is None
    else dict(in_channels=3, out_channels=19))

model = torch.hub.load('bernardomig/neural', args.model, force_reload=True, **kwargs)

# Load model
if args.state_dict is not None:
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
