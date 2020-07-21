import argparse
from collections import OrderedDict
import numpy as np

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

from pytorch_lightning.metrics.functional import confusion_matrix

from pprint import pprint
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--state_dict', type=str, required=True)
args = parser.parse_args()


# Define device
device = torch.device('cuda')

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

model.eval()
torch.set_grad_enabled(False)

metrics = []

for x, y in tqdm(val_loader):
    x = x.to(device)
    y = y.to(device)

    y_pred = torch.argmax(model(x), 1)

    miou = confusion_matrix(y_pred[y != 255], y[y != 255], num_classes=19)
    metrics.append(miou)


cm = sum(metrics)

iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

metrics = {
    # 'accuracy': state.metrics['accuracy'],
    'miou': iou.mean(),
    'iou': iou,
    # 'iou': {name: state.metrics['iou'][id].item() for id, name in enumerate(classes)},
}

pprint(metrics)
