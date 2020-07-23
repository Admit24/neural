from math import log
import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


__all__ = ['ohem_loss', 'OHEMLoss']


def ohem_loss(y_pred, y, ignore_index=-100, thresh=-log(0.7), num_frac=1/16):
    n_min = int(y[y != ignore_index].numel() * num_frac)
    loss = F.cross_entropy(y_pred, y, ignore_index=ignore_index, reduction='none')
    loss = loss.flatten()
    hard = loss[loss > thresh]
    if hard.numel() < n_min:
        return loss.topk(n_min)[0]
    else:
        return hard.mean()


class OHEMLoss(_Loss):

    def __init__(self, ignore_index=-100, thresh=-log(0.7), num_frac=1/16):
        super().__init__()
        self.thresh = thresh
        self.ignore_index = ignore_index
        self.num_frac = num_frac

    def forward(self, y_pred, y):
        return ohem_loss(y_pred, y,
                         ignore_index=self.ignore_index,
                         thresh=self.thresh,
                         num_frac=self.num_frac)
