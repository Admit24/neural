
from neural.losses import OHEMLoss
from neural.optim.lr_scheduler import PolyLR
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
    create_lr_scheduler_with_warmup, LRScheduler)
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from neural.models.segmentation.lednet import LedNet
from neural.engines.segmentation import (
    create_segmentation_trainer, create_segmentation_evaluator)
from neural.data.cityscapes import Cityscapes


from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=768)
parser.add_argument('--state_dict', type=str, required=False)
parser.add_argument('--checkpoint', type=str, required=False)
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
    albu.RandomScale([0.5, 2.0], interpolation=cv2.INTER_CUBIC, p=1.),
    albu.RandomCrop(crop_size, crop_size),
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

model = LedNet(3, 19)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=True)

model = model.to(device)

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=args.learning_rate,
#     weight_decay=args.weight_decay,
# )
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=0.9,
    nesterov=True,
)

if args.checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


# weight = torch.tensor([2.8149201869965,
#                        6.9850029945374,
#                        3.7890393733978,
#                        9.9428062438965,
#                        9.7702074050903,
#                        9.5110931396484,
#                        10.311357498169,
#                        10.026463508606,
#                        4.6323022842407,
#                        9.5608062744141,
#                        7.8698215484619,
#                        9.5168733596802,
#                        10.373730659485,
#                        6.6616044044495,
#                        10.260489463806,
#                        10.287888526917,
#                        10.289801597595,
#                        10.405355453491,
#                        10.138095855713])
# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
loss_fn = OHEMLoss(ignore_index=255, numel_frac=1/16)
loss_fn = loss_fn.cuda()


scheduler = LRScheduler(PolyLR(optimizer, args.learning_rate, total_steps=args.epochs * len(train_loader) - 1000))
scheduler = create_lr_scheduler_with_warmup(
    scheduler, 0, args.learning_rate, 1000)

original_optimizer = optimizer

model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
if args.checkpoint:
    amp.load_state_dict(checkpoint['amp'])

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
    ProgressBar(persist=True).attach(evaluator, ['miou', 'accuracy'])


@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(engine):
    evaluator.run(val_loader)


if local_rank == 0:
    @evaluator.on(Events.COMPLETED)
    def log_results(engine):
        epoch = trainer.state.epoch
        metrics = engine.state.metrics
        miou, accuracy = metrics['miou'], metrics['accuracy']

        lr = original_optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}]: miou={miou}, accuracy={accuracy}, lr={lr}')


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('checkpoints', 'lednet'),
        filename_prefix='lednet',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={
            'model': model if not args.distributed else model.module,
            'optimizer': optimizer,
            'amp': amp,
        },
    )

trainer.run(train_loader, max_epochs=args.epochs)
