from functools import partial

from torch import nn

__all__ = ['IntermediateGrabber', 'OutputGrabber', 'InputGrabber']


class IntermediateGrabber(nn.Module):

    def __init__(self, module, layers, type="output"):
        super(IntermediateGrabber, self).__init__()

        if type not in {'output', 'input'}:
            raise ValueError("type of Intermediate Grabber "
                             "can only be input or output. Got {}."
                             .format(type))

        self.type = type
        self.module = module
        self.layers = layers

    def forward(self, input):
        output = self.module(input)
        return output, self.outputs

    def __enter__(self):
        if isinstance(self.layers, dict):
            self.outputs = {}
        elif isinstance(self.layers, list):
            self.outputs = [None, ] * len(self.layers)
        else:
            raise RuntimeError(
                "the layer spicification has to be an instance of "
                "a dict, or a list.")

        self._hooks = []

        grabber_hook = (output_grabber_hook if self.type == "output" else input_grabber_hook)

        if isinstance(self.layers, dict):
            for name, layer in self.layers.items():
                hook_fn = partial(grabber_hook,
                                  outputs=self.outputs,
                                  id=name)
                hook = layer.register_forward_hook(hook_fn)
                self._hooks.append(hook)
        else:
            for id, layer in enumerate(self.layers):
                hook_fn = partial(grabber_hook,
                                  outputs=self.outputs,
                                  id=id)
                hook = layer.register_forward_hook(hook_fn)
                self._hooks.append(hook)

        return self

    def __exit__(self, _type, _value, _tb):
        for hook in self._hooks:
            hook.remove()
        del self._hooks


OutputGrabber = partial(IntermediateGrabber, type="output")

InputGrabber = partial(IntermediateGrabber, type="input")


def output_grabber_hook(_module, _input, output, outputs, id):
    outputs[id] = output


def input_grabber_hook(_module, input, _output, outputs, id):
    outputs[id] = input
