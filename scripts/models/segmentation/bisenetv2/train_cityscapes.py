
from torch.nn import functional as F
import cv2
from torch import nn
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
    create_lr_scheduler_with_warmup, LRScheduler)
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation.bisenetv2 import bisenetv2, Classifier
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.cityscapes import Cityscapes

from neural.losses import OHEMLoss
from neural.optim.lr_scheduler import PolyLR
from neural.nn.util import DeepSupervision

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)


# SCHEDULERS

class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio
##


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=2e-4)
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
    albu.RandomScale([0.75, 2], interpolation=cv2.INTER_CUBIC, always_apply=True),
    albu.RandomCrop(1024, 512),
    albu.HorizontalFlip(),
    # albu.HueSaturationValue(),
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

model = bisenetv2(3, 19)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

model = DeepSupervision(model, {
    model.semantic.stem: Classifier(16, 19, 128),
    model.semantic.stage3: Classifier(32, 19, 128),
    model.semantic.stage4: Classifier(64, 19, 128),
    model.semantic.stage5[-2]: Classifier(128, 19, 128),
})

model = model.to(device)


optimizer = torch.optim.SGD(
    [
        {'params': (p for p in model.parameters() if p.dim == 1), 'weight_decay': 0},
        {'params': (p for p in model.parameters() if p.dim != 1)},
    ],
    lr=args.learning_rate,
    momentum=0.9,
    weight_decay=5e-4,
)


ohem_fn = OHEMLoss(ignore_index=255)
ohem_fn = ohem_fn.cuda()


def loss_fn(y_pred, y):
    output, aux_outputs = y_pred

    aux_outputs = [F.interpolate(o, size=y.shape[1:], mode='bilinear', align_corners=True) for o in aux_outputs]

    return ohem_fn(output, y) + sum((ohem_fn(o, y) for o in aux_outputs))


scheduler = WarmupPolyLrScheduler(optimizer, 0.9, max_iter=len(train_loader) * args.epochs, warmup_iter=1000)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)


trainer = create_segmentation_trainer(
    model, optimizer, loss_fn,
    device=device,
    use_f16=True,
)


@trainer.on(Events.ITERATION_COMPLETED)
def update_scheduler(_engine):
    scheduler.step()


evaluator = create_segmentation_evaluator(
    model,
    device=device,
    num_classes=19,
)

if local_rank == 0:
    ProgressBar(persist=False).attach(trainer, ['loss'])
    ProgressBar(persist=False).attach(evaluator)


@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(engine):
    evaluator.run(val_loader)


if local_rank == 0:
    @evaluator.on(Events.COMPLETED)
    def log_results(engine):
        epoch = trainer.state.epoch
        metrics = engine.state.metrics
        miou, accuracy = metrics['miou'], metrics['accuracy']
        lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch}]: miou={miou}, accuracy={accuracy}, lr={lr}')


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'bisenetv2'),
        filename_prefix='bisenetv2',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={
            'model': model if not args.distributed else model.module,
        },
    )

trainer.run(train_loader, max_epochs=args.epochs)
