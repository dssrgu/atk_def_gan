import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

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
                 y_dim=5,
                 adv=True,
                 ):
        super(Generator, self).__init__()

        self.adv = adv

        input_c = 256

        y_decoder_lis = [
            nn.ConvTranspose2d(y_dim, input_c, kernel_size=8, stride=1, padding=0, output_padding=0, bias=True),
            nn.BatchNorm2d(input_c),
            nn.ReLU(),
        ]

        if adv:
            input_c *= 2

        bottle_neck_lis = [
            ResnetBlock(input_c, input_c),
            ResnetBlock(input_c, input_c),
            ResnetBlock(input_c, input_c),
        ]

        decoder_lis = [
            nn.ConvTranspose2d(input_c, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        ]


        tanh_lis = [
            nn.Tanh(),
        ]

        self.y_decoder = nn.Sequential(*y_decoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        self.tanh = nn.Sequential(*tanh_lis)

    def forward(self, z, y=None):

        if self.adv:
            y = self.y_decoder(y)
            z = torch.cat([z, y], dim=1)

        z = self.bottle_neck(z)
        z = self.decoder(z)

        if self.adv:
            z = self.tanh(z)

        return z


class Encoder(nn.Module):
    def __init__(self,
                 en_input_nc=3,
                 ):
        super(Encoder, self).__init__()

        encoder_lis = [
            # cifar10: 3*32*32
            nn.Conv2d(en_input_nc, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]

        bottle_neck_lis = [
            ResnetBlock(128, 256),
            ResnetBlock(256, 256),
            ResnetBlock(256, 256),
            ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)

        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias)
        self.rescale = in_dim != out_dim

        project_lis = [
            nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(out_dim),
        ]

        self.project = nn.Sequential(*project_lis)

    def build_conv_block(self, in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
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

        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(out_dim),
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

        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        if self.rescale:
            out = self.project(x) + self.conv_block(x)
        else:
            out = x + self.conv_block(x)
        return out


if __name__ == '__main__':
    encoder = Encoder()
    adv_generator = Generator(adv=True)
    def_generator = Generator(adv=False)

    summary(encoder, (3, 32, 32))
    #summary(adv_generator, (256, 8, 8), (10, 1, 1))
    summary(def_generator, (256, 8, 8))

