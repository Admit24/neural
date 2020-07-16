from collections import OrderedDict
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.engine import Events
from ignite.contrib.handlers import LRScheduler

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model)


from neural.models.segmentation.lednet import lednet
from neural.data.imagenet import Imagenet
from neural.engines.classification import create_classification_evaluator, create_classification_trainer

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--state_dict', type=str, required=False)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda')


distributed = args.distributed
world_size, world_rank, local_rank = setup_distributed(
    distributed, args.local_rank)


train_tfms = albu.Compose([
    albu.Resize(256, 256),
    albu.RandomScale([0.2, 1]),
    albu.RandomCrop(224, 224),
    albu.HorizontalFlip(),
    albu.HueSaturationValue(),
    albu.Normalize(),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.Resize(256, 256),
    albu.CenterCrop(224, 224),
    albu.Normalize(),
    ToTensor(),
])

dataset_dir = get_datasets_root('imagenet')
train_dataset = Imagenet(dataset_dir, split='train', transforms=train_tfms)
val_dataset = Imagenet(dataset_dir, split='val', transforms=val_tfms)


sampler_args = dict(world_size=world_size,
                    local_rank=local_rank,
                    enable=distributed)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=8,
    sampler=create_sampler(train_dataset, **sampler_args),
    shuffle=not distributed,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    sampler=create_sampler(val_dataset, training=False, **sampler_args),
)

model = lednet(3, 19)
model = nn.Sequential(OrderedDict([
    ('encoder', model.encoder),
    ('classifier', nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(128, 1000),
    ))
]))

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=0.9,
)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


scheduler = LRScheduler(torch.optim.lr_scheduler.OneCycleLR(
    optimizer, args.learning_rate, steps_per_epoch=len(train_loader), epochs=args.epochs))


model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


trainer = create_classification_trainer(
    model, optimizer, loss_fn,
    device=device,
    use_f16=True,
)


trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

evaluator = create_classification_evaluator(model, device=device)

if local_rank == 0:
    ProgressBar(persist=False).attach(trainer, ['loss'])
    ProgressBar(persist=False).attach(evaluator)


@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(engine):
    evaluator.run(val_loader)


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'lednet-pretrain'),
        filename_prefix='lednet_pretrain',
        score_name='accuracy',
        score_function=lambda engine: engine.state.metrics['accuracy'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
