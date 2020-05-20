import torch
from torch.nn import functional as F


def swish(input, inplace=False):
    return input * torch.sigmoid(input)


def mish(input, inplace=False):
    return input * torch.tanh(F.softplus(input))


def hard_swish(input, inplace=False):
    return input * F.relu6(input + 3.).div(6.)


def hard_sigmoid(input, inplace=False):
    return F.relu6(input + 3).div(6.)
