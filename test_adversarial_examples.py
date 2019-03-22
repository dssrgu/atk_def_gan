import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from pgd_attack import PGD

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

# load encoder & generators
enc_path = './models/enc_epoch_60.pth'
enc = models.Encoder(en_input_nc).to(device)
enc.load_state_dict(torch.load(enc_path))
enc.eval()

advG_path = './models/advG_epoch_60.pth'
advG = models.Generator(image_nc).to(device)
advG.load_state_dict(torch.load(advG_path))
advG.eval()

defG_path = './models/defG_epoch_60.pth'
defG = models.Generator(image_nc, False).to(device)
defG.load_state_dict(torch.load(defG_path))
defG.eval()

# load PGD attack
pgd = PGD(target_model, defG, device)

def tester(dataset, dataloader):
    num_correct_adv = 0
    num_correct_pgd = 0
    num_correct_def_adv = 0
    num_correct_def_nat = 0
    num_correct_def_pgd = 0
    num_correct_nat = 0
    for i, data in enumerate(dataloader, 0):
        # load images
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        # prep images
        adv_noise = advG(enc(test_img))
        adv_img = adv_noise * eps + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        
        def_adv_noise = defG(enc(adv_img))
        def_adv_img = def_adv_noise + adv_img
        def_adv_img = torch.clamp(def_adv_img, 0, 1)

        def_nat_noise = defG(enc(test_img))
        def_nat_img = def_nat_noise + test_img
        def_nat_img = torch.clamp(def_nat_img, 0, 1)

        pgd_img = pgd.perturb(test_img, test_label)
        
        def_pgd_noise = defG(enc(pgd_img))
        def_pgd_img = def_pgd_noise + pgd_img
        def_pgd_img = torch.clamp(def_pgd_img, 0, 1)

        # calculate acc.
        pred_adv = torch.argmax(target_model(adv_img),1)
        pred_pgd = torch.argmax(target_model(pgd_img), 1)
        pred_def_adv = torch.argmax(target_model(def_adv_img),1)
        pred_def_nat = torch.argmax(target_model(def_nat_img),1)
        pred_def_pgd = torch.argmax(target_model(def_pgd_img), 1)
        pred_nat = torch.argmax(target_model(test_img),1)
        
        num_correct_adv += torch.sum(pred_adv==test_label,0)
        num_correct_pgd += torch.sum(pred_pgd==test_label,0)
        num_correct_def_adv += torch.sum(pred_def_adv==test_label,0)
        num_correct_def_nat += torch.sum(pred_def_nat==test_label,0)
        num_correct_def_pgd += torch.sum(pred_def_pgd==test_label,0)
        num_correct_nat += torch.sum(pred_nat==test_label,0)

    print('num_correct(adv): ', num_correct_adv.item())
    print('num_correct(pgd): ', num_correct_pgd.item())
    print('num_correct(def(adv)): ', num_correct_def_adv.item())
    print('num_correct(def(nat)): ', num_correct_def_nat.item())
    print('num_correct(def(pgd)): ', num_correct_def_pgd.item())
    print('num_correct(nat): ', num_correct_nat.item())
    
    print('accuracy of adv imgs: %f'%(num_correct_adv.item()/len(dataset)))
    print('accuracy of pgd imgs: %f'%(num_correct_pgd.item()/len(dataset)))
    print('accuracy of def(adv) imgs: %f'%(num_correct_def_adv.item()/len(dataset)))
    print('accuracy of def(nat) imgs: %f'%(num_correct_def_nat.item()/len(dataset)))
    print('accuracy of def(pgd) imgs: %f'%(num_correct_def_pgd.item()/len(dataset)))
    print('accuracy of nat imgs: %f'%(num_correct_nat.item()/len(dataset)))
    
    print()

# test adversarial examples in MNIST training dataset
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

print('MNIST training dataset:')
tester(mnist_dataset, train_dataloader)

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

print('MNIST test dataset:')
tester(mnist_dataset_test, test_dataloader)

