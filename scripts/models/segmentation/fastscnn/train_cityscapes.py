
import cv2
import argparse
import os
from logging import info

import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import (
    create_lr_scheduler_with_warmup, CosineAnnealingScheduler, LRScheduler, )
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation.fastscnn import fastscnn, Classifier
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.cityscapes import Cityscapes, MEAN, STD
from neural.nn.util import DeepSupervision

from neural.losses import OHEMLoss
from neural.optim.lr_scheduler import PolyLR

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=768)
parser.add_argument('--state_dict', type=str, required=False)

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
    albu.Normalize(
        mean=MEAN,
        std=STD,
    ),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.Normalize(
        mean=MEAN,
        std=STD,
    ),
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

model = fastscnn(3, 19)
model = DeepSupervision(model, {
    model.downsample: nn.Sequential(
        Classifier(64, 19),
        nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
    ),
    model.features: nn.Sequential(
        Classifier(128, 19),
        nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),
    ),
})


if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.module.load_state_dict(state_dict, strict=True)


model = model.to(device)

weight_params = []
bn_params = []

for m in model.modules():
    if isinstance(m, nn.Conv2d) and m.groups == 1:
        weight_params += list(m.parameters())
    elif isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
        bn_params += list(m.parameters())

num_total_parameters = sum((p.numel() for p in model.parameters()))
num_bn_params = sum((p.numel() for p in bn_params))
num_weight_params = sum((p.numel() for p in weight_params))
assert (num_total_parameters == num_bn_params + num_weight_params), \
    f"{num_total_parameters} != {num_bn_params} + {num_weight_params} ({num_bn_params + num_weight_params})"

optimizer = torch.optim.SGD([
    {'params': weight_params, 'weight_decay': args.weight_decay},
    {'params': bn_params}],
    lr=args.learning_rate,
    momentum=0.85,
    nesterov=True,
)

ohem_fn = OHEMLoss(ignore_index=255).cuda()


def supervised_loss_fn(y_pred, y):
    y_pred, aux_y_pred = y_pred
    return \
        ohem_fn(y_pred, y) \
        + sum((ohem_fn(y_pred, y) for y_pred in aux_y_pred))


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.learning_rate,
    total_steps=args.epochs * len(train_loader),
)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


trainer = create_segmentation_trainer(
    model, optimizer, supervised_loss_fn,
    device=device,
    use_f16=True,
)


@trainer.on(Events.ITERATION_COMPLETED)
def update_scheduler(engine):
    scheduler.step()


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
    @evaluator.on(Events.COMPLETED)
    def log_results(engine):
        epoch = trainer.state.epoch
        metrics = engine.state.metrics
        miou, accuracy = metrics['miou'], metrics['accuracy']

        print(f'Epoch [{epoch}]: miou={miou}, accuracy={accuracy}')


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'fastscnn-weights'),
        filename_prefix='fastscnn',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={
            'wrapped': model if not args.distributed else model.module,
        },
    )

trainer.run(train_loader, max_epochs=args.epochs)
