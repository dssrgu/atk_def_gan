import torch.nn as nn
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


def name_maker(args):
    model_name = 'E{}'.format(args.E_lr) + \
        '_advG{}'.format(args.advG_lr) + \
        '_defG{}'.format(args.defG_lr) + \
        '_targeted' + \
        '_{}'.format(args.seeds) + '/'

    return model_name


def num_correct(output, target):
    '''
    pred = torch.sigmoid(output) >= 0.5
    truth = target >= 0.5
    num_correct = pred.eq(truth).sum()
    '''
    _, pred = torch.max(output, dim=1)
    num_correct = pred.eq(target).sum()
    return num_correct
