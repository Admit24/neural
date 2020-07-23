from math import log
import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


__all__ = ['ohem_loss', 'OHEMLoss']


def ohem_loss(input, target, ignore_index=-100, thresh_loss=-log(0.7), numel_frac=1 / 16):
    n_min = numel_frac * target[target != ignore_index].numel()
    loss = F.cross_entropy(
        input, target, ignore_index=ignore_index, reduction='none')
    hard = loss[loss > thresh_loss]

    if hard.numel() < n_min:
        return loss.topk(n_min)[0].mean()
    else:
        return torch.mean(hard)


class OHEMLoss(_Loss):

    def __init__(self, ignore_index=-100, thresh_loss=-log(0.7), numel_frac=1/16):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh_loss = thresh_loss
        self.numel_frac = numel_frac

    def forward(self, input, target):
        return ohem_loss(input, target,
                         ignore_index=self.ignore_index,
                         thresh_loss=self.thresh_loss,
                         numel_frac=self.numel_frac)
