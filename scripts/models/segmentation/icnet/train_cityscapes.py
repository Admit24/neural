
from itertools import chain
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

from neural.models.segmentation.icnet import icnet_resnet50
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.cityscapes import Cityscapes, MEAN, STD
from neural.nn.util import DeepSupervision

from neural.losses import OHEMLoss

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-4)
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
    albu.RandomScale([-0.25, 1.0], interpolation=cv2.INTER_CUBIC, always_apply=True),
    albu.RandomCrop(712, 712),
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

model = icnet_resnet50(3, 19)
model = DeepSupervision(model, {
    model.head.cff24.lowres: nn.Conv2d(128, 19, 1),
    model.head.cff12.lowres: nn.Conv2d(128, 19, 1),
})


if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)


model = model.to(device)


def non_wd_params(params):
    for p in params:
        if len(p.shape) == 1:
            yield p


def wd_params(params):
    for p in params:
        if len(p.shape) != 1:
            yield p


pretrained_parameters = model.module.encoder.context.parameters()
non_pretrained_parameters = chain(
    model.auxiliary.parameters(),
    model.module.encoder.spatial.parameters(),
    model.module.head.parameters(),
    model.module.classifier.parameters()
)

optimizer = torch.optim.SGD(
    [
        # encoder parameters
        {'params': wd_params(pretrained_parameters),
         'weight_decay': args.weight_decay},
        {'params': non_wd_params(pretrained_parameters)},
        {'params': wd_params(non_pretrained_parameters),
         'weight_decay': args.weight_decay},
        {'params': non_wd_params(non_pretrained_parameters), }
    ],
    lr=args.learning_rate,
    momentum=0.9,
)

# ohem_fn = OHEMLoss(ignore_index=255).cuda()
# loss_fn = ohem_fn


class_freq = torch.from_numpy(Cityscapes.CLASS_FREQ).float()
weight = 1 / torch.log(1.02 + class_freq)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
# loss_fn = OHEMLoss(ignore_index=255)
loss_fn = loss_fn.cuda()


def aux_loss(y_pred, y):
    from torch.nn.functional import interpolate
    y_pred = interpolate(y_pred, size=y.shape[1:], mode='bilinear', align_corners=True)
    return loss_fn(y_pred, y)


def supervised_loss_fn(y_pred, y):
    y_pred, aux_y_pred = y_pred
    return \
        loss_fn(y_pred, y) \
        + 0.4 * sum((aux_loss(y_pred, y) for y_pred in aux_y_pred))


scheduler1 = CosineAnnealingScheduler(
    optimizer, param_name='lr',
    start_value=args.learning_rate / 10, end_value=args.learning_rate / 10 * 1e-4,
    cycle_size=args.epochs * len(train_loader) - 1000,
    param_group_index=0,
)
scheduler1 = create_lr_scheduler_with_warmup(scheduler1, 0, args.learning_rate / 10, 1000)
scheduler2 = CosineAnnealingScheduler(
    optimizer, param_name='lr',
    start_value=args.learning_rate / 10, end_value=args.learning_rate / 10 * 1e-4,
    cycle_size=args.epochs * len(train_loader) - 1000,
    param_group_index=1,
)
scheduler2 = create_lr_scheduler_with_warmup(scheduler2, 0, args.learning_rate / 10, 1000)
scheduler3 = CosineAnnealingScheduler(
    optimizer, param_name='lr',
    start_value=args.learning_rate, end_value=args.learning_rate * 1e-4,
    cycle_size=args.epochs * len(train_loader)-1000,
    param_group_index=2,
)
scheduler3 = create_lr_scheduler_with_warmup(scheduler3, 0, args.learning_rate, 1000)
scheduler4 = CosineAnnealingScheduler(
    optimizer, param_name='lr',
    start_value=args.learning_rate, end_value=args.learning_rate * 1e-4,
    cycle_size=args.epochs * len(train_loader)-1000,
    param_group_index=3,
)
scheduler4 = create_lr_scheduler_with_warmup(scheduler4, 0, args.learning_rate, 1000)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


trainer = create_segmentation_trainer(
    model, optimizer, supervised_loss_fn,
    device=device,
    use_f16=True,
)

trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler1)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler2)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler3)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler4)


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
        dirname=os.path.join('checkpoints', 'icnet-weights'),
        filename_prefix='icnet_resnet50',
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
