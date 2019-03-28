import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
import argparse
from tensorboardX import SummaryWriter

use_cuda = True
image_nc = 1
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1
eps = 0.3

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()

parser.add_argument('--log_base_dir', default='mnist/data', type=str)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--Eadv', default='True', type=boolean_string)
parser.add_argument('--Gadv', default='True', type=boolean_string)
parser.add_argument('--logging', default='False', type=boolean_string)
parser.add_argument('--epochs', default=100, type=int)

args = parser.parse_args()
for arg in vars(args):
    print (arg, getattr(args, arg))

model_name = 'beta{}'.format(args.beta) + ('_Eadv' if args.Eadv else '') + ('_Gadv' if args.Gadv else '') +  '/'

# tensorboard writer
if args.logging:
    log_base_dir = args.log_base_dir + '/' +  model_name
    writer = SummaryWriter(log_base_dir)
else:
    writer = None

# Define what device we are using print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./MNIST_target_model.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          args.beta,
                          args.Eadv,
                          args.Gadv,
                          BOX_MIN,
                          BOX_MAX,
                          eps,
                          model_name,
                          writer)

advGAN.train(dataloader, args.epochs)
