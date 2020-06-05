import torch
from torch.jit import trace
import argparse

from neural.models.segmentation import contextnet

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="the name of the contextnet model.",
                    choices=['contextnet12', 'contextnet14', 'contextnet18'])
parser.add_argument('--state_dict', required=True,
                    help=("the state dict containing the parameters of "
                          "the model to be exported."),
                    type=str)
parser.add_argument('--output_file', required=True,
                    help="the output torchscript file.",
                    type=str)
parser.add_argument('--width', default=512,
                    help=("the input width of the model."))
parser.add_argument('--height', default=512,
                    help=("the input height of the model."))
parser.add_argument('--input_channels', required=False,
                    default=3,
                    help=("the number of input channels. "
                          "By default, for a rgb image this is 3."))
parser.add_argument('--num_classes', required=False,
                    default=19,
                    help=("the number of output classes. "
                          "By default, this is 1000 to match the"
                          "imagenet dataset."))
args = parser.parse_args()


Model = getattr(contextnet, args.model)
model = Model(args.input_channels, args.num_classes)

width, height = args.width, args.height
example_input = torch.randn((1, args.input_channels, height, width))

compiled_model = trace(model.eval(), example_input)
compiled_model.save(args.output_file)
