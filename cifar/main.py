import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from wideresnet import WideResNet
import argparse
from tensorboardX import SummaryWriter
from utils import boolean_string, name_maker, normalizer
import os

use_cuda = True
image_nc = 3
model_num_labels = 10
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1
eps = 0.3
pgd_iter = [1]
# model save path
models_path = './models/'
# image output path
out_path = './out/'

parser = argparse.ArgumentParser()

parser.add_argument('--log_base_dir', default='mnist/data', type=str)
parser.add_argument('--seeds', default=0, type=int)
parser.add_argument('--E_lr', default=0.01, type=float)
parser.add_argument('--advG_lr', default=0.001, type=float)
parser.add_argument('--defG_lr', default=0.1, type=float)
parser.add_argument('--logging', default='False', type=boolean_string)
parser.add_argument('--overwrite', default='True', type=boolean_string)
parser.add_argument('--epochs', default=100, type=int)

args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

model_name = name_maker(args)

# tensorboard writer
if args.logging:
    log_base_dir = args.log_base_dir + '/' + model_name
    print("logging at:", log_base_dir)
    
    # overwrite?
    if os.path.exists(log_base_dir) and not args.overwrite:
        print()
        print('result already exists!')
        exit()
    
    writer = SummaryWriter(log_base_dir)

else:
    writer = None


if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(models_path + model_name):
    os.makedirs(models_path + model_name)

# Define what device we are using print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./model_best.pth.tar"
targeted_model = WideResNet().to(device)
checkpoint = torch.load(pretrained_model, map_location=device)
targeted_model.load_state_dict(checkpoint['state_dict'])
targeted_model.eval()
normalizer = normalizer()

# MNIST train dataset and dataloader declaration
cifar_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=normalizer, download=True)
dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Attack(device,
                       targeted_model,
                       model_num_labels,
                       image_nc,
                       BOX_MIN,
                       BOX_MAX,
                       eps,
                       pgd_iter,
                       models_path,
                       out_path,
                       model_name,
                       writer,
                       args.E_lr,
                       args.advG_lr,
                       args.defG_lr)

advGAN.train(dataloader, args.epochs)
