from torch.nn import Module
from neural.nn.functional import swish, mish, hard_swish, hard_sigmoid


class Swish(Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return swish(input, self.inplace)


class Mish(Module):
    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return mish(input, self.inplace)


class HardSwish(Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return hard_swish(input, self.inplace)


class HardSigmoid(Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return hard_sigmoid(input, self.inplace)
