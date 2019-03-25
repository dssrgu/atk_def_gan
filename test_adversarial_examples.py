import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import models
from models import MNIST_target_net
from pgd_attack import PGD
import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', default=60, type=int)
parser.add_argument('--model_name', default='', type=str)

args = parser.parse_args()

use_cuda=True
image_nc=1
batch_size = 128
eps = 0.3

en_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
print()

# load the pretrained model
model_path = "./MNIST_target_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(model_path))
target_model.eval()

epoch=args.epoch
model_name = (args.model_name+'_') if args.model_name else args.model_name

# load encoder & generators
enc_path = './models/' + model_name + 'enc_epoch_{}.pth'.format(epoch)
enc = models.Encoder(en_input_nc).to(device)
enc.load_state_dict(torch.load(enc_path))
enc.eval()

advG_path = './models/' + model_name + 'advG_epoch_{}.pth'.format(epoch)
advG = models.Generator(image_nc).to(device)
advG.load_state_dict(torch.load(advG_path))
advG.eval()

defG_path = './models/' + model_name + 'defG_epoch_{}.pth'.format(epoch)
defG = models.Generator(image_nc, adv=False).to(device)
defG.load_state_dict(torch.load(defG_path))
defG.eval()

# load PGD attack
pgd = PGD(target_model, enc, defG, device)

def tester(dataset, dataloader, save_img=False):
    num_correct_adv = 0
    num_correct_pgd = 0
    num_correct_def_adv = 0
    num_correct_def = 0
    num_correct_def_pgd = 0
    num_correct = 0

    test_img_full = []
    adv_img_full = []
    def_img_full = []
    def_adv_img_full = []
    def_pgd_img_full = []

    for i, data in enumerate(dataloader, 0):
        # load images
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        target_labels = torch.randint_like(test_label, 0, 10)
        target_one_hot = torch.eye(10, device=device)[target_labels]
        target_one_hot = target_one_hot.view(-1, 10, 1, 1)
        
        # prep images
        adv_noise = advG(enc(test_img), target_one_hot)
        adv_img = adv_noise * eps + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        
        def_adv_noise = defG(enc(adv_img))
        def_adv_img = def_adv_noise + adv_img
        def_adv_img = torch.clamp(def_adv_img, 0, 1)

        def_noise = defG(enc(test_img))
        def_img = def_noise + test_img
        def_img = torch.clamp(def_img, 0, 1)

        pgd_img = pgd.perturb(test_img, test_label)
        
        def_pgd_noise = defG(enc(pgd_img))
        def_pgd_img = def_pgd_noise + pgd_img
        def_pgd_img = torch.clamp(def_pgd_img, 0, 1)

        # calculate acc.
        pred_adv = torch.argmax(target_model(adv_img),1)
        pred_pgd = torch.argmax(target_model(pgd_img), 1)
        pred_def_adv = torch.argmax(target_model(def_adv_img),1)
        pred_def = torch.argmax(target_model(def_img),1)
        pred_def_pgd = torch.argmax(target_model(def_pgd_img), 1)
        pred = torch.argmax(target_model(test_img),1)
        
        num_correct_adv += torch.sum(pred_adv==test_label,0)
        num_correct_pgd += torch.sum(pred_pgd==test_label,0)
        num_correct_def_adv += torch.sum(pred_def_adv==test_label,0)
        num_correct_def += torch.sum(pred_def==test_label,0)
        num_correct_def_pgd += torch.sum(pred_def_pgd==test_label,0)
        num_correct += torch.sum(pred==test_label,0)

        if save_img and i < 1:
            test_img_full.append(test_img)
            adv_img_full.append(adv_img)
            def_img_full.append(def_img)
            def_adv_img_full.append(def_adv_img)
            def_pgd_img_full.append(def_pgd_img)

    print('num_correct(adv): ', num_correct_adv.item())
    print('num_correct(pgd): ', num_correct_pgd.item())
    print('num_correct(def(adv)): ', num_correct_def_adv.item())
    print('num_correct(def(nat)): ', num_correct_def.item())
    print('num_correct(def(pgd)): ', num_correct_def_pgd.item())
    print('num_correct(nat): ', num_correct.item())
    
    print('accuracy of adv imgs: %f'%(num_correct_adv.item()/len(dataset)))
    print('accuracy of pgd imgs: %f'%(num_correct_pgd.item()/len(dataset)))
    print('accuracy of def(adv) imgs: %f'%(num_correct_def_adv.item()/len(dataset)))
    print('accuracy of def(nat) imgs: %f'%(num_correct_def.item()/len(dataset)))
    print('accuracy of def(pgd) imgs: %f'%(num_correct_def_pgd.item()/len(dataset)))
    print('accuracy of nat imgs: %f'%(num_correct.item()/len(dataset)))

    l_inf = np.amax(np.abs(def_pgd_img.cpu().detach().numpy()-test_img.cpu().detach().numpy()))
    print('l-inf of def(pgd) imgs:%f'%(l_inf))
    
    print()

    
    if save_img:
        test_img_full = torch.cat(test_img_full)
        adv_img_full = torch.cat(adv_img_full)
        def_img_full = torch.cat(def_img_full)
        def_adv_img_full = torch.cat(def_adv_img_full)
        def_pgd_img_full = torch.cat(def_pgd_img_full)
        
        test_grid = make_grid(test_img_full)
        adv_grid = make_grid(adv_img_full)
        def_grid = make_grid(def_img_full)
        def_adv_grid = make_grid(def_adv_img_full)
        def_pgd_grid = make_grid(def_pgd_img_full)

        save_image(test_grid, './out/test_grid.png')
        save_image(adv_grid, './out/adv_grid.png')
        save_image(def_grid, './out/def_grid.png')
        save_image(def_adv_grid, './out/def_adv_grid.png')
        save_image(def_pgd_grid, './out/def_pgd_grid.png')

        print('images saved')
    

# test adversarial examples in MNIST training dataset
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

print('MNIST training dataset:')
tester(mnist_dataset, train_dataloader)

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

print('MNIST test dataset:')
tester(mnist_dataset_test, test_dataloader, save_img=True)

