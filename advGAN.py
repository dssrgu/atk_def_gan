import torch.nn as nn
import torch
#import numpy as np
import models
import torch.nn.functional as F
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
                 model_name,
                 log_base_dir,
                 writer,
                 init_lr=0.001):
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
        self.log_base_dir = log_base_dir
        self.writer = writer
        self.init_lr = init_lr

        self.en_input_nc = image_nc
        self.E = models.Encoder(self.en_input_nc).to(device)
        self.advG = models.Generator(image_nc, vec_nc).to(device)
        self.defG = models.Generator(image_nc, adv=False).to(device)
        self.mine = models.Mine(image_nc, vec_nc).to(device)

        # initialize all weights
        self.E.apply(weights_init)
        self.advG.apply(weights_init)
        self.defG.apply(weights_init)
        self.mine.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                              lr=self.init_lr)
        self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                            lr=self.init_lr)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                            lr=self.init_lr)
        self.optimizer_mine = torch.optim.Adam(self.mine.parameters(),
                                               lr=self.init_lr)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    # generate images for training
    def gen_images(self, x, z):

        z_res = z.view(-1, self.vec_nc, 1, 1)
        # make adv image
        adv_noise = self.advG(self.E(x), z_res)
        adv_images = adv_noise * self.eps + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # make def(adv) image
        def_adv_images = self.defG(self.E(adv_images)) + adv_images
        def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

        # make def(nat) image
        def_images = self.defG(self.E(x)) + x
        def_images = torch.clamp(def_images, self.box_min, self.box_max)

        return adv_noise, adv_images, def_adv_images, def_images

    def mi_est(self, adv_noise, z, z_bar):

        l_bound = torch.mean(self.mine(adv_noise, z)) - torch.log(torch.mean(torch.exp(self.mine(adv_noise, z_bar))))

        return l_bound

    def train_batch(self, x, labels):

        # optimize E
        for i in range(1):

            # clear grad
            self.optimizer_E.zero_grad()

            # sample z from uniform dist.
            z = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1

            _, adv_images, def_adv_images, def_images = self.gen_images(x, z)

            # adv loss
            logits_adv = self.model(adv_images)
            loss_adv = -F.cross_entropy(logits_adv, labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            # ???
            loss_E = loss_adv + loss_def

            loss_E.backward()

            self.optimizer_E.step()

        # optimize advG
        for i in range(1):

            # clear grad
            self.optimizer_advG.zero_grad()

            # sample z from uniform dist.
            z = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1
            z_bar = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1

            adv_noise, adv_images, def_adv_images, _ = self.gen_images(x, z)

            # adv loss
            logits_adv = self.model(adv_images)
            loss_adv = -F.cross_entropy(logits_adv, labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = -F.cross_entropy(logits_def_adv, labels)

            # backprop
            loss_advG = loss_adv + loss_def_adv

            loss_advG.backward(retain_graph=True)
            
            # gradient checker
            adv_norm = 0
            for p in self.advG.parameters():
                param_norm = p.grad.data.norm(2)
                adv_norm += param_norm.item() ** 2
            adv_norm = adv_norm ** (1. / 2)

            self.optimizer_advG.step()

            #clear grad
            self.optimizer_advG.zero_grad()

            # MI loss
            mi_loss = -self.mi_est(adv_noise, z, z_bar)

            mi_loss.backward()

            # gradient checker
            mi_norm = 0
            for p in self.advG.parameters():
                param_norm = p.grad.data.norm(2)
                mi_norm += param_norm.item() ** 2
            mi_norm = mi_norm ** (1. / 2)

            # mi gradient regularizer
            min_norm = min(adv_norm, mi_norm)

            grad_reg = min_norm / mi_norm

            for p in self.advG.parameters():
                p.grad.data.mul_(grad_reg)

            self.optimizer_advG.step()

        # optimize defG
        for i in range(1):
            
            # clear grad
            self.optimizer_defG.zero_grad()

            # sample z from uniform dist.
            z = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1

            _, _, def_adv_images, def_images = self.gen_images(x, z)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            loss_defG = loss_def_adv + loss_def
            
            loss_defG.backward()
            self.optimizer_defG.step()

        # optimize Mine
        for i in range(1):

            # clear grad
            self.optimizer_mine.zero_grad()

            # sample z from uniform dist.
            z = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1
            z_bar = torch.rand(labels.shape[0], self.vec_nc).to(self.device) * 2 - 1

            adv_noise, adv_images, _, _ = self.gen_images(x, z)

            # MI loss
            loss_mine = -self.mi_est(adv_noise, z, z_bar)

            loss_mine.backward()
        
            # gradient checker
            mi_norm = 0
            for p in self.mine.parameters():
                param_norm = p.grad.data.norm(2)
                mi_norm += param_norm.item() ** 2
            mi_norm = mi_norm ** (1. / 2)
            
            self.optimizer_mine.step()

        return grad_reg, mi_norm, torch.sum(loss_E).item(), torch.sum(loss_advG).item(),\
               torch.sum(loss_defG).item(), loss_mine.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=self.init_lr/10)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=self.init_lr/10)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=self.init_lr/10)
                self.optimizer_mine = torch.optim.Adam(self.mine.parameters(),
                                                       lr=self.init_lr/10)
            if epoch == 80:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=self.init_lr/100)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=self.init_lr/100)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=self.init_lr/100)
                self.optimizer_mine = torch.optim.Adam(self.mine.parameters(),
                                                       lr=self.init_lr/100)

            loss_E_sum = 0
            loss_advG_sum = 0
            loss_defG_sum = 0
            loss_mine_sum = 0
            reg_sum = 0
            mi_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                reg_batch, mi_batch, loss_E_batch, loss_advG_batch, loss_defG_batch, loss_mine_batch = \
                    self.train_batch(images, labels)
                loss_E_sum += loss_E_batch
                loss_advG_sum += loss_advG_batch
                loss_defG_sum += loss_defG_batch
                loss_mine_sum += loss_mine_batch
                reg_sum += reg_batch
                mi_sum += mi_batch


            # print statistics
            num_batch = len(train_dataloader)
            print('grad_reg: %.5f' % (reg_sum/num_batch))
            print('mi_norm: %.5f' % (mi_sum/num_batch))
            print("epoch %d:\nloss_E: %.5f, loss_advG: %.5f, loss_defG: %.5f, loss_mine: %.5f\n" %
                  (epoch, loss_E_sum/num_batch, loss_advG_sum/num_batch,
                   loss_defG_sum/num_batch, loss_mine_sum/num_batch))

            # write to tensorboard
            self.writer.add_scalar(self.log_base_dir+'/grad_reg_vec{}'.format(self.vec_nc), reg_sum/num_batch, epoch)
            self.writer.add_scalar(self.log_base_dir+'/mi_norm_vec{}'.format(self.vec_nc), mi_sum/num_batch, epoch)
            self.writer.add_scalar(self.log_base_dir+'/loss_E_vec{}'.format(self.vec_nc), loss_E_sum/num_batch, epoch)
            self.writer.add_scalar(self.log_base_dir+'/loss_advG_vec{}'.format(self.vec_nc), loss_advG_sum/num_batch, epoch)
            self.writer.add_scalar(self.log_base_dir+'/loss_defG_vec{}'.format(self.vec_nc), loss_defG_sum/num_batch, epoch)
            self.writer.add_scalar(self.log_base_dir+'/loss_mine_vec{}'.format(self.vec_nc), loss_mine_sum/num_batch, epoch)

            # save generator
            if epoch%20==0:
                E_file_name = models_path + self.model_name + 'E_epoch_' + str(epoch) + '.pth'
                advG_file_name = models_path + self.model_name + 'advG_epoch_' + str(epoch) + '.pth'
                defG_file_name = models_path + self.model_name + 'defG_epoch_' + str(epoch) + '.pth'
                mine_file_name = models_path + self.model_name + 'mine_epoch_' + str(epoch) + '.pth'
                torch.save(self.E.state_dict(), E_file_name)
                torch.save(self.advG.state_dict(), advG_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)
                torch.save(self.mine.state_dict(), mine_file_name)

        self.writer.close()

