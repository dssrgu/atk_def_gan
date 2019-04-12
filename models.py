import torch
import torch.nn as nn
import torch.nn.functional as F


# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(64*7*7, 10)

    def forward(self, x):
        assert not torch.any(x > 1)
        assert not torch.any(x < 0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        return x


# not used
class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 adv=True,
                 ):
        super(Generator, self).__init__()

        self.adv = adv

        input_c = 128

        decoder_lis = [
            nn.ConvTranspose2d(input_c, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
        ]

        tanh_lis = [
            nn.Tanh(),
        ]

        self.decoder = nn.Sequential(*decoder_lis)
        self.tanh = nn.Sequential(*tanh_lis)

    def forward(self, z):

        z = self.decoder(z)

        if self.adv:
            z = self.tanh(z)

        return z


class Encoder(nn.Module):
    def __init__(self,
                 en_input_nc,
                 ):
        super(Encoder, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(en_input_nc, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_lis)

    def forward(self, x):
        x = self.encoder(x)

        return x


class Mine(nn.Module):
    def __init__(self,
                 en_input_nc,
                 z_dim=128,
                 ):
        super(Mine, self).__init__()

        img_lis = [
            # MNIST:1*28*28
            nn.Conv2d(en_input_nc, 1, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # 1*14*14
        ]

        z_lis = [
            # z: z_dim*7*7
            nn.Conv2d(z_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # 1*7*7
        ]

        full_lis = [
            nn.Linear(14*14+7*7, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        ]

        self.img_encoder = nn.Sequential(*img_lis)
        self.z_encoder = nn.Sequential(*z_lis)
        self.full_encoder = nn.Sequential(*full_lis)

    def forward(self, x, z):
        x = self.img_encoder(x)
        x = x.view(-1, 14*14)
        z = self.z_encoder(z)
        z = z.view(-1, 7*7)
        concat = torch.cat([x, z], dim=1)
        out = self.full_encoder(concat)
        return out


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
