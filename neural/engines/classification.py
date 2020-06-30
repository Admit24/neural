from ignite.engine import Engine, Events


def create_classification_trainer(
        model, optimizer, loss_fn,
        device, use_f16=True, non_blocking=True
):
    from ignite.engine import create_supervised_trainer
    from ignite.metrics import RunningAverage, Loss

    def _prepare_batch(batch, device, non_blocking):
        image = batch['image'].to(device, non_blocking=non_blocking)
        label = batch['label'].to(device, non_blocking=non_blocking)
        return image, label

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()

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


def create_classification_evaluator(
    model,
    device,
    non_blocking=True,
):
    from ignite.metrics import Accuracy, TopKCategoricalAccuracy
    from ignite.engine import create_supervised_evaluator

    def _prepare_batch(batch, device, non_blocking):
        image = batch['image'].to(device, non_blocking=non_blocking)
        label = batch['label'].to(device, non_blocking=non_blocking)
        return image, label

    metrics = {
        'accuracy': Accuracy(),
        'top5': TopKCategoricalAccuracy(k=5),
    }

    evaluator = create_supervised_evaluator(
        model, metrics, device,
        non_blocking=non_blocking,
        prepare_batch=_prepare_batch)

    return evaluator
