import torch.nn as nn
import torch
#import numpy as np
import models
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision
import os

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 vec_nc,
                 box_min,
                 box_max,
                 eps,
                 model_name):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.vec_nc = vec_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.eps = eps
        self.model_name = (model_name + '_') if model_name else model_name

        self.en_input_nc = image_nc
        self.E = models.Encoder(self.en_input_nc).to(device)
        self.advG = models.Generator(image_nc, vec_nc).to(device)
        self.defG = models.Generator(image_nc, vec_nc, adv=False).to(device)
        self.D = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.E.apply(weights_init)
        self.advG.apply(weights_init)
        self.defG.apply(weights_init)
        self.D.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                              lr=0.001)
        self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                            lr=0.001)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    # generate images for training
    def gen_images(self, x, labels):

        target_labels = torch.randint_like(labels, 0, self.model_num_labels)
        target_one_hot = torch.eye(self.model_num_labels, device=self.device)[target_labels]
        target_one_hot = target_one_hot.view(-1, self.model_num_labels, 1, 1)

        # make adv image
        adv_images = self.advG(self.E(x), target_one_hot) * self.eps + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # make def(adv) image
        def_adv_images = self.defG(self.E(adv_images)) + adv_images
        def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

        # make def(nat) image
        def_images = self.defG(self.E(x)) + x
        def_images = torch.clamp(def_images, self.box_min, self.box_max)

        return target_labels, adv_images, def_adv_images, def_images

    def train_batch(self, x, labels):

        D_loss = nn.BCELoss()

        true_vec = Variable(torch.FloatTensor(labels.size(0), 1).to(self.device).fill_(1.0), requires_grad=False)
        false_vec = Variable(torch.FloatTensor(labels.size(0), 1).to(self.device).fill_(0.0), requires_grad=False)

        # optimize D
        for i in range(1):

            #clear grad
            self.optimizer_D.zero_grad()

            _, adv_images, def_adv_images, def_images = self.gen_images(x, labels)


            loss_D = D_loss(self.D(x), true_vec)
            loss_D += D_loss(self.D(adv_images), false_vec)
            loss_D += D_loss(self.D(def_adv_images), false_vec)
            loss_D += D_loss(self.D(def_images), false_vec)

            loss_D /= 4

            loss_D.backward()
            self.optimizer_D.step()

        # optimize E
        for i in range(1):

            #clear grad
            self.optimizer_E.zero_grad()

            target_labels, adv_images, def_adv_images, def_images = self.gen_images(x, labels)

            # adv loss
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, target_labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            # D loss
            loss_D = D_loss(self.D(adv_images), true_vec)
            loss_D += D_loss(self.D(def_adv_images), true_vec)
            loss_D += D_loss(self.D(def_images), true_vec)
            loss_D /= 3


            #loss_E = loss_adv + loss_def_adv + loss_def
            loss_E = loss_D

            loss_E.backward()
            self.optimizer_E.step()

        # optimize advG
        for i in range(1):

            # clear grad
            self.optimizer_advG.zero_grad()

            target_labels, adv_images, def_adv_images, _ = self.gen_images(x, labels)

            # adv loss
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, target_labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, target_labels)

            # D loss
            loss_D = D_loss(self.D(def_adv_images), true_vec)

            # backprop
            #loss_advG = loss_adv + loss_def_adv
            loss_advG = loss_def_adv + loss_D

            loss_advG.backward()
            self.optimizer_advG.step()

        # optimize defG
        for i in range(1):
            
            # clear grad
            self.optimizer_defG.zero_grad()

            _, _, def_adv_images, def_images = self.gen_images(x, labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            # D loss
            loss_D += D_loss(self.D(def_adv_images), true_vec)
            loss_D += D_loss(self.D(def_images), true_vec)
            loss_D /= 2

            loss_defG = loss_def_adv + loss_def + loss_D
            
            loss_defG.backward()
            self.optimizer_defG.step()

        return torch.sum(loss_D).item(), torch.sum(loss_E).item(), torch.sum(loss_advG).item(), torch.sum(loss_defG).item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                    lr=0.0001)
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=0.0001)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=0.0001)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                    lr=0.00001)
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=0.00001)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=0.00001)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=0.00001)

            loss_D_sum = 0
            loss_E_sum = 0
            loss_advG_sum = 0
            loss_defG_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_E_batch, loss_advG_batch, loss_defG_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_E_sum += loss_E_batch
                loss_advG_sum += loss_advG_batch
                loss_defG_sum += loss_defG_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_E: %.3f, loss_advG: %.3f, loss_defG: %.3f,\n" %
                  (epoch, loss_D_sum/num_batch, loss_E_sum/num_batch, loss_advG_sum/num_batch, loss_defG_sum/num_batch))

            # save generator
            if epoch%20==0:
                D_file_name = models_path + self.model_name + 'D_epoch_' + str(epoch) + '.pth'
                E_file_name = models_path + self.model_name + 'E_epoch_' + str(epoch) + '.pth'
                advG_file_name = models_path + self.model_name + 'advG_epoch_' + str(epoch) + '.pth'
                defG_file_name = models_path + self.model_name + 'defG_epoch_' + str(epoch) + '.pth'

                torch.save(self.D.state_dict(), D_file_name)
                torch.save(self.E.state_dict(), E_file_name)
                torch.save(self.advG.state_dict(), advG_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)

