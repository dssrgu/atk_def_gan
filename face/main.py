import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from wideresnet import WideResNet
import argparse
from tensorboardX import SummaryWriter
from utils import boolean_string, name_maker, load_dataset, load_models
import os

use_cuda = True
image_nc = 3
model_num_labels = 5
BOX_MIN = 0
BOX_MAX = 1
pgd_iter = [1]
# model save path
models_path = './models/'
# image output path
out_path = './out/'

parser = argparse.ArgumentParser()

parser.add_argument('--log_base_dir', default='face/data', type=str)
parser.add_argument('--seeds', default=0, type=int)
parser.add_argument('--E_lr', default=0.01, type=float)
parser.add_argument('--advG_lr', default=0.001, type=float)
parser.add_argument('--defG_lr', default=0.1, type=float)
parser.add_argument('--logging', default='False', type=boolean_string)
parser.add_argument('--overwrite', default='True', type=boolean_string)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--data', default='./dataset/real_and_fake_face')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--epsilon', default=8/255, type=float)

args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

model_name = name_maker(args)

# tensorboard writer
if args.logging:
    log_base_dir = args.log_base_dir + '/' + model_name
    print("logging at:", log_base_dir)
    writer = SummaryWriter(log_base_dir)
else:
    writer = None

if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(models_path + model_name):
    os.makedirs(models_path + model_name)
else:
    # overwrite?
    if not args.overwrite and len(os.listdir(models_path + model_name)) != 0:
        print()
        print('result already exists!')
        exit()

# Define what device we are using print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./Face_target_model2.pth"
targeted_model, _ = load_models(device)
checkpoint = torch.load(pretrained_model, map_location=device)
targeted_model.load_state_dict(checkpoint)
targeted_model.eval()

# FACE train dataset and dataloader declaration
#cifar_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transforms.ToTensor(), download=True)
#dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
train_loader, _ = load_dataset(args.data, batch_size=args.batch_size, seed=args.seed)
advGAN = AdvGAN_Attack(device,
                       targeted_model,
                       model_num_labels,
                       image_nc,
                       BOX_MIN,
                       BOX_MAX,
                       args.epsilon,
                       pgd_iter,
                       models_path,
                       out_path,
                       model_name,
                       writer,
                       args.E_lr,
                       args.advG_lr,
                       args.defG_lr)

advGAN.train(train_loader, args.epochs)
