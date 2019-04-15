import torch.nn as nn
import torchvision.transforms as transforms
import torch


# for slurm script compatibility
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
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
        '_{}'.format(args.seeds) + '/'

    return model_name


# image normalizer
def normalizer():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])

    return transform

