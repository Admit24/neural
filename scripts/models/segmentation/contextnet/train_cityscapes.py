
import cv2
import argparse
import os

import torch
from torch.utils.data import DataLoader

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import (
    create_lr_scheduler_with_warmup, CosineAnnealingScheduler)
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation.contextnet import ContextNet
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.cityscapes import Cityscapes

from neural.losses import OHEMLoss

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=768)
parser.add_argument('--state_dict', type=str, required=False)
parser.add_argument('--width_multiplier', type=int, default=1)

parser.add_argument('--distributed', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

distributed = args.distributed
world_size, world_rank, local_rank = setup_distributed(
    distributed, args.local_rank)

device = torch.device('cuda')

crop_size = args.crop_size


train_tfms = albu.Compose([
    albu.RandomScale([-0.5, 1.0], interpolation=cv2.INTER_CUBIC, always_apply=True),
    albu.RandomCrop(512, 1024),
    albu.HorizontalFlip(),
    albu.HueSaturationValue(),
    albu.Normalize(),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])

dataset_dir = get_datasets_root('cityscapes')
train_dataset = Cityscapes(dataset_dir, split='train', transforms=train_tfms)
val_dataset = Cityscapes(dataset_dir, split='val', transforms=val_tfms)


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

model = ContextNet(3, 19,
                   scale_factor=4,
                   width_multiplier=args.width_multiplier)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)


model = model.to(device)

optimizer = torch.optim.AdamW(
    [
        {'params': (p for p in model.parameters() if len(p.shape) != 1),
         'weight_decay': args.weight_decay, },
        {'params': (p for p in model.parameters() if len(p.shape) == 1), },
    ],
    lr=args.learning_rate,
)

loss_fn = OHEMLoss(ignore_index=255)
loss_fn = loss_fn.cuda()


scheduler = CosineAnnealingScheduler(
    optimizer, 'lr',
    args.learning_rate, args.learning_rate / 1000,
    args.epochs * len(train_loader) - 1000,
)
scheduler = create_lr_scheduler_with_warmup(
    scheduler, 0, args.learning_rate, 1000)


model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


trainer = create_segmentation_trainer(
    model, optimizer, loss_fn,
    device=device,
    use_f16=True,
)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)


evaluator = create_segmentation_evaluator(
    model,
    device=device,
    num_classes=19,
)

if local_rank == 0:
    ProgressBar(persist=True).attach(trainer, ['loss'])
    ProgressBar(persist=True).attach(evaluator)


@trainer.on(Events.EPOCH_COMPLETED(every=5))
def evaluate(engine):
    evaluator.run(val_loader)


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'contextnet14'),
        filename_prefix='contextnet',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
