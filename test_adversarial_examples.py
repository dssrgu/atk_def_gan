import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import models
import os
from models import MNIST_target_net
from pgd_attack import PGD
import argparse
import numpy as np
from utils import boolean_string, parameters_count, name_maker

use_cuda = True
image_nc = 1
batch_size = 128
models_path = './models/'
out_path = './out/'


# train on a given data set
def tester(dataset, dataloader, device, target_model, E, defG, advG, eps, out_path, model_name, label_count=True, save_img=False):
    
    # load PGD
    pgd = PGD(target_model, E, defG, device)

    num_correct_adv = 0
    num_correct_pgd = 0
    num_correct_def_adv = 0
    num_correct_def = 0
    num_correct_def_pgd = 0
    num_correct_def_pgd_nat = 0
    num_correct = 0

    test_img_full = []
    adv_img_full = []
    pgd_img_full = []
    pgd_nat_img_full = []
    def_img_full = []
    def_adv_img_full = []
    def_pgd_img_full = []
    def_pgd_nat_img_full = []

    pred_adv_full = []
    pred_pgd_full = []
    pred_def_pgd_full = []
    pred_def_pgd_nat_full = []

    for i, data in enumerate(dataloader, 0):
        # load images
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        # prep images
        x_encoded = E(test_img)

        adv_noise = advG(x_encoded)
        adv_img = adv_noise * eps + test_img
        adv_img = torch.clamp(adv_img, 0, 1)

        def_adv_noise = defG(E(adv_img))
        def_adv_img = def_adv_noise + adv_img
        def_adv_img = torch.clamp(def_adv_img, 0, 1)

        def_noise = defG(E(test_img))
        def_img = def_noise + test_img
        def_img = torch.clamp(def_img, 0, 1)

        pgd_img = pgd.perturb(test_img, test_label, itr=1)

        def_pgd_noise = defG(E(pgd_img))
        def_pgd_img = def_pgd_noise + pgd_img
        def_pgd_img = torch.clamp(def_pgd_img, 0, 1)

        pgd_nat_img = pgd.perturb(test_img, test_label, itr=0)

        def_pgd_nat_noise = defG(E(pgd_nat_img))
        def_pgd_nat_img = def_pgd_nat_noise + pgd_nat_img
        def_pgd_nat_img = torch.clamp(def_pgd_nat_img, 0, 1)

        # calculate acc.
        pred = torch.argmax(target_model(test_img), 1)
        pred_adv = torch.argmax(target_model(adv_img), 1)
        pred_pgd = torch.argmax(target_model(pgd_nat_img), 1)
        pred_def_adv = torch.argmax(target_model(def_adv_img), 1)
        pred_def = torch.argmax(target_model(def_img), 1)
        pred_def_pgd = torch.argmax(target_model(def_pgd_img), 1)
        pred_def_pgd_nat = torch.argmax(target_model(def_pgd_nat_img), 1)

        num_correct += torch.sum(pred == test_label, 0)
        num_correct_adv += torch.sum(pred_adv == test_label, 0)
        num_correct_pgd += torch.sum(pred_pgd == test_label, 0)
        num_correct_def_adv += torch.sum(pred_def_adv == test_label, 0)
        num_correct_def += torch.sum(pred_def == test_label, 0)
        num_correct_def_pgd += torch.sum(pred_def_pgd == test_label, 0)
        num_correct_def_pgd_nat += torch.sum(pred_def_pgd_nat == test_label, 0)

        if label_count:
            pred_adv_full.append(pred_adv)
            pred_pgd_full.append(pred_pgd)
            pred_def_pgd_full.append(pred_def_pgd)
            pred_def_pgd_nat_full.append(pred_def_pgd_nat)

        if save_img and i < 1:
            test_img_full.append(test_img)
            adv_img_full.append(adv_img)
            pgd_img_full.append(pgd_img)
            pgd_nat_img_full.append(pgd_nat_img)
            def_img_full.append(def_img)
            def_adv_img_full.append(def_adv_img)
            def_pgd_img_full.append(def_pgd_img)
            def_pgd_nat_img_full.append(def_pgd_nat_img)

    print('num_correct(nat): ', num_correct.item())
    print('num_correct(adv): ', num_correct_adv.item())
    print('num_correct(pgd): ', num_correct_pgd.item())
    print('num_correct(def(adv)): ', num_correct_def_adv.item())
    print('num_correct(def(nat)): ', num_correct_def.item())
    print('num_correct(def(pgd)): ', num_correct_def_pgd.item())
    print('num_correct(def(pgd_nat)): ', num_correct_def_pgd_nat.item())
    print()

    print('accuracy of nat imgs: %f' % (num_correct.item() / len(dataset)))
    print('accuracy of adv imgs: %f' % (num_correct_adv.item() / len(dataset)))
    print('accuracy of pgd imgs: %f' % (num_correct_pgd.item() / len(dataset)))
    print('accuracy of def(adv) imgs: %f' % (num_correct_def_adv.item() / len(dataset)))
    print('accuracy of def(nat) imgs: %f' % (num_correct_def.item() / len(dataset)))
    print('accuracy of def(pgd) imgs: %f' % (num_correct_def_pgd.item() / len(dataset)))
    print('accuracy of def(pgd_nat) imgs: %f' % (num_correct_def_pgd_nat.item() / len(dataset)))
    print()

    l_inf = np.amax(np.abs(adv_img.cpu().detach().numpy() - test_img.cpu().detach().numpy()))
    print('l-inf of adv imgs:%f' % (l_inf))
    l_inf = np.amax(np.abs(def_img.cpu().detach().numpy() - test_img.cpu().detach().numpy()))
    print('l-inf of def imgs:%f' % (l_inf))
    l_inf = np.amax(np.abs(def_adv_img.cpu().detach().numpy() - test_img.cpu().detach().numpy()))
    print('l-inf of def(adv) imgs:%f' % (l_inf))
    l_inf = np.amax(np.abs(def_pgd_img.cpu().detach().numpy() - test_img.cpu().detach().numpy()))
    print('l-inf of def(pgd) imgs:%f' % (l_inf))
    l_inf = np.amax(np.abs(def_pgd_nat_img.cpu().detach().numpy() - test_img.cpu().detach().numpy()))
    print('l-inf of def(pgd_nat) imgs:%f' % (l_inf))

    print()

    if label_count:
        pred_adv_full = torch.cat(pred_adv_full)
        preds = pred_adv_full.cpu().detach().numpy()
        print('label counts in adv imgs:')
        print(np.unique(preds, return_counts=True))
        
        pred_pgd_full = torch.cat(pred_pgd_full)
        preds = pred_pgd_full.cpu().detach().numpy()
        print('label counts in pgd imgs:')
        print(np.unique(preds, return_counts=True))
        
        pred_def_pgd_full = torch.cat(pred_def_pgd_full)
        preds = pred_def_pgd_full.cpu().detach().numpy()
        print('label counts in def_pgd imgs:')
        print(np.unique(preds, return_counts=True))
        
        pred_def_pgd_nat_full = torch.cat(pred_def_pgd_nat_full)
        preds = pred_def_pgd_nat_full.cpu().detach().numpy()
        print('label counts in def_pgd_nat imgs:')
        print(np.unique(preds, return_counts=True))
        
        print()

    if save_img:
        test_img_full = torch.cat(test_img_full)
        adv_img_full = torch.cat(adv_img_full)
        pgd_img_full = torch.cat(pgd_img_full)
        pgd_nat_img_full = torch.cat(pgd_nat_img_full)
        def_img_full = torch.cat(def_img_full)
        def_adv_img_full = torch.cat(def_adv_img_full)
        def_pgd_img_full = torch.cat(def_pgd_img_full)
        def_pgd_nat_img_full = torch.cat(def_pgd_nat_img_full)

        test_grid = make_grid(test_img_full)
        adv_grid = make_grid(adv_img_full)
        pgd_grid = make_grid(pgd_img_full)
        pgd_nat_grid = make_grid(pgd_nat_img_full)
        def_grid = make_grid(def_img_full)
        def_adv_grid = make_grid(def_adv_img_full)
        def_pgd_grid = make_grid(def_pgd_img_full)
        def_pgd_nat_grid = make_grid(def_pgd_nat_img_full)

        if not os.path.exists(out_path + model_name):
            os.makedirs(out_path + model_name)

        save_image(test_grid, out_path + model_name + 'test_grid.png')
        save_image(adv_grid, out_path + model_name + 'adv_grid.png')
        save_image(pgd_grid, out_path + model_name + 'pgd_grid.png')
        save_image(pgd_nat_grid, out_path + model_name + 'pgd_nat_grid.png')
        save_image(def_grid, out_path + model_name + 'def_grid.png')
        save_image(def_adv_grid, out_path + model_name + 'def_adv_grid.png')
        save_image(def_pgd_grid, out_path + model_name + 'def_pgd_grid.png')
        save_image(def_pgd_nat_grid, out_path + model_name + 'def_pgd_nat_grid.png')

        print('images saved')


# train on both training and test set
def test_full(device, target_model, E, defG, advG, eps, out_path, model_name, label_count=True, save_img=True):
    
    # test adversarial examples in MNIST training dataset
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print('MNIST training dataset:')
    tester(mnist_dataset, train_dataloader, device, target_model, E, defG, advG, eps, out_path, model_name, label_count, False)

    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(),
                                                    download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

    print('MNIST test dataset:')
    tester(mnist_dataset_test, test_dataloader, device, target_model, E, defG, advG, eps, out_path, model_name, label_count, save_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_base_dir', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--seeds', default=0, type=int)
    parser.add_argument('--lr_E', default=0.001, type=float)
    parser.add_argument('--lr_advG', default=0.001, type=float)
    parser.add_argument('--lr_defG', default=0.001, type=float)
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--parameters_count', action='store_true')
    parser.add_argument('--labels_count', action='store_true')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    model_name = name_maker(args)

    en_input_nc = image_nc
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print()

    # load the pretrained model
    model_path = "./MNIST_target_model.pth"
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(model_path, map_location=device))
    if args.parameters_count:
        print('number of parameters(model):', parameters_count(target_model))
    target_model.eval()

    epoch = args.epoch

    # load encoder & generators
    E_path = models_path + model_name + 'E_epoch_{}.pth'.format(epoch)
    E = models.Encoder(en_input_nc).to(device)
    E.load_state_dict(torch.load(E_path, map_location=device))
    if args.parameters_count:
        print('number of parameters(E):', parameters_count(E))
    E.eval()

    advG_path = models_path + model_name + 'advG_epoch_{}.pth'.format(epoch)
    advG = models.Generator(adv=True).to(device)
    advG.load_state_dict(torch.load(advG_path, map_location=device))
    if args.parameters_count:
        print('number of parameters(advG):', parameters_count(advG))
    advG.eval()

    defG_path = models_path + model_name + 'defG_epoch_{}.pth'.format(epoch)
    defG = models.Generator(adv=False).to(device)
    defG.load_state_dict(torch.load(defG_path, map_location=device))
    if args.parameters_count:
        print('number of parameters(defG):', parameters_count(defG))
        print()
    defG.eval()

    test_full(device, target_model, E, defG, advG, args.eps, out_path, model_name, label_count=True)
