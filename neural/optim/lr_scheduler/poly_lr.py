from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['PolyLR']


class PolyLR(_LRScheduler):
    """Decays the learning rate of each parameter group based on a polynomial equation:

    .. math::
        lr = lr_{initial} \left(1 - \frac{T}{T_0}\right)^{\gamma}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        initial_lr (float or list): The initial learning rate, or a list of initial learning 
            rate for each parameter group.
        total_steps (int): The total number of iterations. Depends if this scheduler is 
            stepped each epoch or at each iteration.
        gamma (float): The polynomial factor. Default: 0.9. 
        last_epoch (int): The index of the last epoch. Default: -1.

    """

    def __init__(self, optimizer, initial_lr, total_steps, gamma=0.9, last_epoch=-1):

        self.initial_lr = (
            [initial_lr] * len(optimizer.param_groups)
            if not isinstance(initial_lr, (list, tuple))
            else list(initial_lr)
        )
        self.total_steps = total_steps
        self.gamma = gamma

        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            lr * (1 - self.last_epoch / self.total_steps)**self.gamma
            for lr in self.initial_lr
        ]
