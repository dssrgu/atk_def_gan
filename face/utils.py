import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from dis_models.resnet_64 import ResNetAC
from gen_models.resnet_64 import ResNetGenerator

from robustness.datasets import CustomImageNet


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
    ),
    transforms.ToTensor(),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

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
    '''
    x_copy = x.clone()
    batch_size = x_copy.size()[0]
    x_copy = torch.stack([transforms.functional.normalize(x_copy[i], mean=[mu/255.0 for mu in [125.3, 123.0, 113.9]],
                                                          std=[sig/255.0 for sig in [63.0, 62.1, 66.7]]) for i in range(batch_size)])

    '''
    return target_model(x)


def load_dataset(path, batch_size, seed):
    '''
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop([64, 64]),
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
    '''
    groups = [(200, 204), (281, 285), (10, 14), (371, 374), (393, 397)]
    ds = CustomImageNet('/data_large/readonly/ImageNet-Fast/imagenet/', groups)
    ds.transform_train = TRAIN_TRANSFORMS
    ds.transform_test = TEST_TRANSFORMS
    train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)
    return train_loader, val_loader


def load_models(device):
    gen = ResNetGenerator(n_classes=5).to(device)
    dis = ResNetAC(n_classes=5).to(device)

    if torch.cuda.is_available():
        gen = torch.nn.DataParallel(gen)
        dis = torch.nn.DataParallel(dis)
    return dis, gen


def num_correct(output, target):
    '''
    pred = torch.sigmoid(output) >= 0.5
    truth = target >= 0.5
    num_correct = pred.eq(truth).sum()
    '''
    _, pred = torch.max(output, dim=1)
    num_correct = pred.eq(target).sum()
    return num_correct
