import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
import argparse
from tensorboardX import SummaryWriter

use_cuda = True
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--image_nc', default=1, type=int)
parser.add_argument('--vec_nc', default=10, type=int)
parser.add_argument('--eps', default=0.3, type=float)
parser.add_argument('--log_base_dir', default='mnist/data', type=str)

args = parser.parse_args()

for arg in vars(args):
    print (arg, getattr(args, arg))
print()

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./MNIST_target_model.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# tensorboard writer
writer = SummaryWriter()

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          args.image_nc,
                          args.vec_nc,
                          BOX_MIN,
                          BOX_MAX,
                          args.eps,
                          args.model_name,
                          args.log_base_dir,
                          writer,
                       )

advGAN.train(dataloader, args.epochs)
