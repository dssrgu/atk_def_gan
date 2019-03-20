import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
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
                 box_min,
                 box_max,
                 eps):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.eps = eps

        self.gen_input_nc = image_nc
        self.advG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.defG = models.Generator(self.gen_input_nc, image_nc, False).to(device)
        #self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.advG.apply(weights_init)
        self.defG.apply(weights_init)

        # initialize optimizers
        self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                            lr=0.001)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):

        # optimize adv
        for i in range(5):

            # clear grad
            self.optimizer_advG.zero_grad()
            
            # make adv image
            adv_noise = self.advG(x)

            adv_images = adv_noise * self.eps + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            # make def(adv) image
            def_adv_images = self.defG(adv_images) + adv_images
            def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real-other, zeros)
            loss_adv = torch.sum(loss_adv)

            # cal def(adv) loss
            logits_model = self.model(def_adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_def_adv = torch.max(real-other, zeros)
            loss_def_adv = torch.sum(loss_def_adv)

            # backprop
            loss_advG = loss_adv + loss_def_adv
            
            loss_advG.backward()
            self.optimizer_advG.step()

        # optimize def
        for i in range(1):
            
            # clear grad
            self.optimizer_defG.zero_grad()

            # cal 
            adv_noise = self.advG(x)
            adv_images = adv_noise * self.eps + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)
            
            def_adv_images = self.defG(adv_images) + adv_images
            def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)
            
            def_nat_images = self.defG(x) + x
            def_nat_images = torch.clamp(def_nat_images, self.box_min, self.box_max)

            self.optimizer_defG.zero_grad()

            # adv_image loss
            logits_model = self.model(def_adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(other-real, zeros)
            loss_adv = torch.sum(loss_adv)

            # nat_image loss
            logits_model = self.model(def_nat_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_nat = torch.max(other-real, zeros)
            loss_nat = torch.sum(loss_nat)

            loss_defG = loss_adv + loss_nat
            
            loss_defG.backward()
            self.optimizer_defG.step()

        return loss_advG.item(), loss_defG.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=0.0001)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=0.00001)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=0.00001)
            
            loss_advG_sum = 0
            loss_defG_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_advG_batch, loss_defG_batch = \
                    self.train_batch(images, labels)
                loss_advG_sum += loss_advG_batch
                loss_defG_sum += loss_defG_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_advG: %.3f, loss_defG: %.3f,\n" %
                  (epoch, loss_advG_sum/num_batch, loss_defG_sum/num_batch))

            # save generator
            if epoch%20==0:
                advG_file_name = models_path + 'advG_epoch_' + str(epoch) + '.pth'
                defG_file_name = models_path + 'defG_epoch_' + str(epoch) + '.pth'
                torch.save(self.advG.state_dict(), advG_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)

