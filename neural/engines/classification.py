from ignite.engine import Engine, Events


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
