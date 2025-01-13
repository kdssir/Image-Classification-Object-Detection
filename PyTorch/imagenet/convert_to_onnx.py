import torch
import torch.onnx
import argparse
import torchvision.models as models
#resnet18 = models.resnet18()

# A model class instance (class not shown)

parser = argparse.ArgumentParser(description='PyTorch to Onnx Model Conversion')
parser.add_argument('--model_name', dest='model_name', default='Resnet18',type=str,
                    help='give a unique model name to save')

parser.add_argument('--classes', '--num_classes', default=2, type=int,
                    metavar='classes', help='number of classes', dest='classes')

args = parser.parse_args()

model = models.resnet18(num_classes = args.classes)

weights_path = args.model_name + "_best.pth.tar" 
# Load the weights from a file (.pth usually)
state_dict = torch.load(weights_path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items():
    print("key",k)
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
# Load the weights now into a model net architecture defined by our class
#model.load_state_dict(state_dict['state_dict'])

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 3, 112, 112)

torch.onnx.export(model, dummy_input, args.model_name + ".onnx")
