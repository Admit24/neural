import argparse
import os
import torch
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.classification import mobilenetv3
from neural.data.imagenet import Imagenet
from neural.engines import create_classification_evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    choices=['mobilenetv3_small, mobilenetv3_large'])
parser.add_argument('--state_dict', required=True)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

device = torch.device(args.device)

tfms = albu.Compose([
    albu.Resize(256, 256),
    albu.CenterCrop(224, 224),
    albu.Normalize(),
    ToTensor(),
])

dataset_dir = os.path.join(os.environ.get('DATASET_DIR'), 'imagenet')
dataset = Imagenet(root_dir=dataset_dir,
                   split='val',
                   transforms=tfms)


train_loader = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=8)

model = (getattr(vgg, args.model))(3, 1000)
state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict)
model = model.cuda()

evaluator = create_classification_evaluator(model, device=device)
ProgressBar(persist=True).attach(evaluator)


state = evaluator.run(train_loader)

print(state.metrics)
