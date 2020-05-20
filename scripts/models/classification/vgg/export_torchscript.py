import torch
from torch.jit import trace
import argparse

from neural.models.classification import vgg

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="the name of the efficientnet model.",
                    choices=['vgg11', 'vgg13',
                             'vgg16', 'vgg19'])
parser.add_argument('--state_dict', required=True,
                    help=("the state dict containing the parameters of "
                          "the model to be exported."),
                    type=str)
parser.add_argument('--output_file', required=True,
                    help="the output torchscript file.",
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
args = parser.parse_args()

Model = getattr(vgg, args.model)
model = Model(args.input_channels, args.num_classes)

state_dict = torch.load(args.state_dict, map_location='cpu')
model.load_state_dict(state_dict)

input_size = 224 if args.input_size is None else args.input_size

example_input = torch.randn((1, args.input_channels, input_size, input_size))

compiled_model = trace(model.eval(), example_input)

compiled_model.save(args.output_file)
