from torch import nn

from neural.nn.util import OutputGrabber

__all__ = ['DeepSupervision']


class DeepSupervision(nn.Module):
    def __init__(self, module, auxiliary_modules):
        super().__init__()
        self.module = module
        self.layers = [layer for layer, _ in auxiliary_modules]
        self.auxiliary = nn.ModuleList([
            module for _, module in auxiliary_modules
        ])

    def forward(self, input):
        if self.training:
            with OutputGrabber(self.module, self.layers) as grabber:
                output, aux_outputs = grabber(input)

            return output, [
                module(output)
                for module, output in zip(self.auxiliary, aux_outputs)
            ]
        else:
            return self.module(input)
