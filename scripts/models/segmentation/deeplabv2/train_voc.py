
import argparse
import os

import torch
from torch.utils.data import DataLoader

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import LRScheduler
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation.deeplab import deeplabv2_resnet18
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.voc2012 import VOC2012Segmentation

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-5)
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
    albu.RandomScale([0.5, 1.5]),
    albu.PadIfNeeded(crop_size, crop_size),
    albu.RandomCrop(crop_size, crop_size),
    albu.HorizontalFlip(),
    albu.HueSaturationValue(),
    albu.Normalize(),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.PadIfNeeded(crop_size, crop_size),
    albu.CenterCrop(crop_size, crop_size),
    albu.Normalize(),
    ToTensor(),
])

dataset_dir = get_datasets_root('PASCAL_VOC2012')
train_dataset = VOC2012Segmentation(dataset_dir, split='train', transforms=train_tfms)
val_dataset = VOC2012Segmentation(dataset_dir, split='val', transforms=val_tfms)


sampler_args = dict(world_size=world_size,
                    local_rank=local_rank,
                    enable=distributed)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=4,
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

model = deeplabv2_resnet18(3, 21)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)


model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=0.9,
)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
loss_fn = loss_fn.to(device)


scheduler = LRScheduler(
    torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.learning_rate,
        epochs=args.epochs, steps_per_epoch=len(train_loader)))


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
    num_classes=21,
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

        print(f'Epoch [{epoch}]: miou={miou}, accuracy={accuracy}')


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'weights'),
        filename_prefix='deeplabv2_resnet18-voc2012',
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
