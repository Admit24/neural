from functools import partial
from ignite.engine import Engine, Events, _prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss
from ignite.metrics.confusion_matrix import (
    ConfusionMatrix,
    mIoU, IoU,
    DiceCoefficient,
    cmAccuracy,
)
try:
    from apex import amp
except:
    from warnings import warn
    warn("apex is not installed in your platform. "
         "Half-precision training or Distributed training will not be possible.")

__all__ = [
    'create_segmentation_trainer',
    'create_segmentation_evaluator',
]


def create_segmentation_trainer(
    model,
    optimizer,
    loss_fn,
    device,
    use_f16=False,
    non_blocking=True,
):
    def update_fn(_trainer, batch):
        model.train()
        optimizer.zero_grad()
        # batch = batch['image'], batch['mask']
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if use_f16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        return loss.item()

    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x) \
        .attach(trainer, 'loss')

    return trainer


def create_segmentation_evaluator(
    model,
    device,
    num_classes=19,
    loss_fn=None,
    non_blocking=True
):

    cm = partial(ConfusionMatrix, num_classes)

    metrics = {
        'iou': IoU(cm()),
        'miou': mIoU(cm()),
        'accuracy': cmAccuracy(cm()),
        'dice': DiceCoefficient(cm()),
    }

    if loss_fn is not None:
        metrics['loss'] = Loss(loss_fn)

    evaluator = create_supervised_evaluator(
        model,
        metrics,
        device,
        non_blocking=non_blocking,
    )

    return evaluator
