import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

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
from neural.data.bdd import BDDSegmentation

from neural.losses import OHEMLoss

from neural.utils.training import (
    setup_distributed, get_datasets_root, create_sampler)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=768)
parser.add_argument('--state_dict', type=str, required=False)

parser.add_argument('--teacher', type=str, required=True)

parser.add_argument('--distributed', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

distributed = args.distributed
world_size, world_rank, local_rank = setup_distributed(
    distributed, args.local_rank)

device = torch.device('cuda')

crop_size = args.crop_size

train_tfms = albu.Compose([
    albu.RandomScale([0.5, 2.0]),
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
train_dataset = BDDSegmentation(
    dataset_dir, split='train', transforms=train_tfms)
val_dataset = BDDSegmentation(dataset_dir, split='val', transforms=val_tfms)


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

student = ContextNet(3, 19)
teacher = ContextNet(3, 19, width_multiplier=2)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    student.load_state_dict(state_dict, strict=True)

state_dict = torch.load(args.teacher, map_location='cpu')
teacher.load_state_dict(state_dict, strict=True)

student = student.to(device)
teacher = teacher.to(device)

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)

supervised_loss_fn = OHEMLoss(ignore_index=255, numel_frac=0.05).cuda()


def distillation_loss_fn(y_student, y_teacher, temperature=1):
    return nn.KLDivLoss()(F.log_softmax(y_student/temperature, dim=1),
                          F.softmax(y_teacher/temperature, dim=1))


scheduler = CosineAnnealingScheduler(
    optimizer, 'lr',
    args.learning_rate, args.learning_rate / 1000,
    args.epochs * len(train_loader),
)
scheduler = create_lr_scheduler_with_warmup(
    scheduler, 0, args.learning_rate, 1000)

(teacher, student), optimizer = amp.initialize(
    [teacher, student], optimizer, opt_level="O2")
if args.distributed:
    student = convert_syncbn_model(student)
    teacher = DistributedDataParallel(teacher)
    student = DistributedDataParallel(student)


def create_segmentation_distillation_trainer(
        student, teacher, optimizer,
        supervised_loss_fn, distillation_loss_fn,
        device, use_f16=True, non_blocking=True):
    from ignite.engine import Engine, Events, _prepare_batch
    from ignite.metrics import RunningAverage, Loss

    def update_fn(_trainer, batch):
        student.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)

        student_pred = student(x)
        with torch.no_grad():
            teacher_pred = teacher(x)

        supervised_loss = supervised_loss_fn(student_pred, y)
        distillation_loss = distillation_loss_fn(teacher_pred, student_pred)

        loss = supervised_loss + distillation_loss

        if use_f16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        return {
            'loss': loss.item(),
            'supervised_loss': supervised_loss.item(),
            'distillation_loss': distillation_loss.item(),
        }

    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x['loss'])    \
        .attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['supervised_loss'])    \
        .attach(trainer, 'supervised_loss')
    RunningAverage(output_transform=lambda x: x['distillation_loss'])    \
        .attach(trainer, 'distillation_loss')

    return trainer


trainer = create_segmentation_distillation_trainer(
    student, teacher, optimizer,
    supervised_loss_fn, distillation_loss_fn,
    device)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

evaluator = create_segmentation_evaluator(
    student, device=device, num_classes=19)

if local_rank == 0:
    ProgressBar(persist=True).attach(
        trainer, ['loss', 'supervised_loss', 'distillation_loss'])
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

        print(f'Epoch [{epoch}]: miou={miou}, accuracy={accuracy}')


if local_rank == 0:
    checkpointer = ModelCheckpoint(
        dirname=os.path.join('contextnet-weights-bdd100k'),
        filename_prefix='contextnet',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={'model': student if not args.distributed else student.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)
