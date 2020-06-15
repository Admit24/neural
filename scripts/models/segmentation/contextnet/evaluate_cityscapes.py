import argparse

import torch
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation import contextnet
from neural.engines.segmentation import create_segmentation_evaluator
from neural.data.cityscapes import Cityscapes

from neural.utils.training import get_datasets_root

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='contextnet14', 
                    choices=['contextnet12', 'contextnet14', 'contextnet18'])
parser.add_argument('--state_dict', type=str, required=True)
parser.add_argument('--width_multiplier', type=int, default=1)
args = parser.parse_args()


#Define device
device = torch.device('cuda')

#Define transformations to Validation set
val_tfms = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])
#Get Validation set of cityscapes
dataset_dir = get_datasets_root('cityscapes')
val_dataset = Cityscapes(dataset_dir, split='val', transforms=val_tfms)


#Load dataset
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)
### Create model

Model = getattr(contextnet, args.model)
model = Model(3, 19, width_multiplier=args.width_multiplier)

### Load model
state_dict = torch.load(args.state_dict, map_location='cpu')
model.load_state_dict(state_dict, strict=True)

model = model.to(device)

evaluator = create_segmentation_evaluator(
    model,
    device=device,
    num_classes=19,
)

state = evaluator.run(val_loader)

print(state.metrics)
