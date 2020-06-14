import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from dis_models.resnet import ResNetAC
from gen_models.resnet import ResNetGenerator


# for slurm script compatibility
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.0002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.0002)
        nn.init.constant_(m.bias.data, 0)


# count number of trainable parameters for a model
def parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# model name maker
def name_maker(args):
    model_name = 'E{}'.format(args.E_lr) + \
        '_advG{}'.format(args.advG_lr) + \
        '_defG{}'.format(args.defG_lr) + \
        '_targeted' + \
        '_low' + \
        '_{}'.format(args.seeds) + '/'

    return model_name


# image normalizer
def normalized_eval(x, target_model, batch_size=128):
    x_copy = x.clone()
    batch_size = x_copy.size()[0]
    x_copy = torch.stack([transforms.functional.normalize(x_copy[i], mean=[mu/255.0 for mu in [125.3, 123.0, 113.9]],
                                                          std=[sig/255.0 for sig in [63.0, 62.1, 66.7]]) for i in range(batch_size)])

    return target_model(x_copy)


def load_dataset(path, batch_size, seed):
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop([128, 128]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    d_set = torchvision.datasets.ImageFolder(
            root=path,
            transform=train_transform,
    )

    val_split = 0.2

    d_size = len(d_set)
    indices = list(range(d_size))
    split = int(np.floor(val_split * d_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size,
                                             sampler=val_sampler)

    return train_loader, val_loader


def load_models():
    gen = ResNetGenerator(n_classes=1)
    dis = ResNetAC(n_classes=1, bn=True)
    return dis, gen
