import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F
from torch import distributed as dist


class AdaptWeightedCE(_WeightedLoss):
    def __init__(self, num_classes, weight_fn, decay=0.99,
                 ignore_index=-100, reduce=None, reduction='mean'):
        super(AdaptWeightedCE, self).__init__(
            weight=torch.zeros(num_classes, dtype=torch.int64),
            reduce=reduce,
            reduction=reduction)

        self.num_classes = num_classes
        self.weight_fn = weight_fn
        self.ignore_index = ignore_index
        self.decay = decay

        self.register_buffer('num_examples', torch.zeros(1, dtype=torch.int64))

    def forward(self, input, target):
        weight = torch.bincount(target[target != self.ignore_index], minlength=self.num_classes)
        num_examples = torch.numel(target[target != self.ignore_index])

        # if dist.is_initialized():
        #     weight = dist.all_reduce(weight)
        #     num_examples = dist.all_reduce(num_examples)

        self.weight = weight + self.decay * self.weight
        self.num_examples = num_examples + self.decay * self.num_examples

        weight = self.weight_fn(self.weight.float() / self.num_examples)

        return F.cross_entropy(input, target, weight=weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
