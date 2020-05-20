import torch
from torch.onnx import export
import argparse

from neural.models.classification import efficientnet

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="the name of the efficientnet model.",
                    choices=['efficientnet_b0', 'efficientnet_b1',
                             'efficientnet_b2', 'efficientnet_b3',
                             'efficientnet_b4', 'efficientnet_b5',
                             'efficientnet_b6', 'efficientnet_b7'])
parser.add_argument('--state_dict', required=True,
                    help=("the state dict containing the parameters of "
                          "the model to be exported."),
                    type=str)
parser.add_argument('--output_file', required=True,
                    help="the output onnx file.",
                    type=str)
parser.add_argument('--input_size', required=False,
                    help=("the input size of the model. By default, the input"
                          "is defined as the paper speficied."))
parser.add_argument('--input_channels', required=False,
                    default=3,
                    help=("the number of input channels. "
                          "By default, for a rgb image this is 3."))
parser.add_argument('--num_classes', required=False,
                    default=1000,
                    help=("the number of output classes. "
                          "By default, this is 1000 to match the"
                          "imagenet dataset."))
parser.add_argument('-v', '--verbose', action='store_true',
                    help="Show logging information")
args = parser.parse_args()


Model = getattr(efficientnet, args.model)
model = Model(args.input_channels, args.num_classes)


if args.input_size is None:
    input_sizes = {
        'efficientnet_b0': 224,
        'efficientnet_b1': 240,
        'efficientnet_b2': 260,
        'efficientnet_b3': 300,
        'efficientnet_b4': 380,
        'efficientnet_b5': 456,
        'efficientnet_b6': 528,
        'efficientnet_b7': 600,
    }
    input_size = input_sizes[args.model]
else:
    input_size = args.input_size

example_input = torch.randn((1, args.input_channels, input_size, input_size))

export(model, example_input, args.output_file,
       verbose=False,
       training=False,
       input_names=['input'],
       output_names=['logits'],
       opset_version=9,
       dynamic_axes={
           'input': {0: 'batch'},
           'logits': {0: 'batch'},
       })
