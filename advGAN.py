import torch.nn as nn
import torch
#import numpy as np
import models
import torch.nn.functional as F
#import torchvision
import os
from test_adversarial_examples import test_full
from pgd_attack import PGD

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
                 eps,
                 model_name,
                 writer,
                 init_lr=0.001):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.eps = eps
        self.model_name = (model_name + '_') if model_name else model_name
        self.writer = writer
        self.init_lr = init_lr
        self.rec_loss = nn.MSELoss()

        self.en_input_nc = image_nc
        self.E = models.Encoder(self.en_input_nc).to(device)
        self.advG = models.Generator(image_nc).to(device)
        self.defG = models.Generator(image_nc, adv=False).to(device)
        self.recG = models.Generator(image_nc).to(device)

        # initialize all weights
        self.E.apply(weights_init)
        self.advG.apply(weights_init)
        self.defG.apply(weights_init)
        self.recG.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                              lr=self.init_lr)
        self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                            lr=self.init_lr)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                            lr=self.init_lr)
        self.optimizer_recG = torch.optim.Adam(self.recG.parameters(),
                                               lr=self.init_lr)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    # generate images for training
    def gen_images(self, x):

        # reconstruct image
        rec_images = self.recG(self.E(x))
        rec_images = torch.clamp(rec_images, self.box_min, self.box_max)

        # make adv image
        adv_images = self.advG(self.E(x)) * self.eps + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # make def(adv) image
        def_adv_images = self.defG(self.E(adv_images)) + adv_images
        def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

        # make def(nat) image
        def_images = self.defG(self.E(x)) + x
        def_images = torch.clamp(def_images, self.box_min, self.box_max)

        return rec_images, adv_images, def_adv_images, def_images

    # performance tester
    def test(self):
        
        self.E.eval()
        self.defG.eval()
        
        test_full(self.device, self.model, self.E, self.defG, self.advG, self.eps, label_count=True, save_img=False)
        
        self.E.train()
        self.defG.train()

    # train helper
    def train_batch(self, x, labels):

        # optimize E
        for i in range(1):

            # clear grad
            self.optimizer_E.zero_grad()

            rec_images, adv_images, def_adv_images, def_images = self.gen_images(x)

            # rec loss
            loss_rec = self.rec_loss(rec_images, x)

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
            loss_E = loss_rec

            loss_E.backward()
            
            self.optimizer_E.step()

        # optimize advG
        for i in range(1):

            # clear grad
            self.optimizer_advG.zero_grad()

            _, adv_images, def_adv_images, _ = self.gen_images(x)

            # adv loss
            logits_adv = self.model(adv_images)
            loss_adv = -F.cross_entropy(logits_adv, labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = -F.cross_entropy(logits_def_adv, labels)

            # backprop
            loss_advG = loss_adv + loss_def_adv

            loss_advG.backward()
            
            self.optimizer_advG.step()

        # optimize defG
        for i in range(1):
            
            # clear grad
            self.optimizer_defG.zero_grad()

            _, _, def_adv_images, def_images = self.gen_images(x)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            loss_defG = loss_def_adv + loss_def
            
            loss_defG.backward()

            self.optimizer_defG.step()

        # optimize recG
        for i in range(1):

            # clear grad
            self.optimizer_recG.zero_grad()

            rec_images, adv_images, _, _ = self.gen_images(x)

            # rec loss

            loss_rec = self.rec_loss(rec_images, x)

            loss_recG = loss_rec

            loss_recG.backward()

            self.optimizer_recG.step()

        # pgd performance check
        
        self.E.eval()
        self.defG.eval()
        
        pgd = PGD(self.model, self.E, self.defG, self.device)
        pgd_img = pgd.perturb(x, labels)
        def_pgd_noise = self.defG(self.E(pgd_img))
        def_pgd_img = def_pgd_noise * self.eps + pgd_img

        pred = torch.argmax(self.model(def_pgd_img), 1)
        num_correct = torch.sum(pred == labels, 0)
        pgd_acc = num_correct.item()/len(labels)

        self.E.train()
        self.defG.train()

        return pgd_acc, torch.sum(loss_E).item(), torch.sum(loss_advG).item(),\
               torch.sum(loss_defG).item(), torch.sum(loss_recG).item()

    # main training function
    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=self.init_lr/10)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=self.init_lr/10)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=self.init_lr/10)
                self.optimizer_recG = torch.optim.Adam(self.recG.parameters(),
                                                       lr=self.init_lr/10)
            if epoch == 80:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                      lr=self.init_lr/100)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                    lr=self.init_lr/100)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                    lr=self.init_lr/100)
                self.optimizer_recG = torch.optim.Adam(self.recG.parameters(),
                                                       lr=self.init_lr/100)

            loss_E_sum = 0
            loss_advG_sum = 0
            loss_defG_sum = 0
            loss_recG_sum = 0
            pgd_acc_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                pgd_acc_batch, loss_E_batch, loss_advG_batch, loss_defG_batch, loss_recG_batch = \
                    self.train_batch(images, labels)
                loss_E_sum += loss_E_batch
                loss_advG_sum += loss_advG_batch
                loss_defG_sum += loss_defG_batch
                loss_recG_sum += loss_recG_batch
                pgd_acc_sum += pgd_acc_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_E: %.5f, loss_advG: %.5f, loss_defG: %.5f, loss_recG: %.5f, pgd_acc: %.5f\n" %
                  (epoch, loss_E_sum/num_batch, loss_advG_sum/num_batch,
                   loss_defG_sum/num_batch, loss_recG_sum/num_batch, pgd_acc_sum/num_batch))

            # write to tensorboard
            if self.writer:
                self.writer.add_scalar('loss_E', loss_E_sum/num_batch, epoch)
                self.writer.add_scalar('loss_advG', loss_advG_sum/num_batch, epoch)
                self.writer.add_scalar('loss_defG', loss_defG_sum/num_batch, epoch)
                self.writer.add_scalar('loss_mine', loss_recG_sum/num_batch, epoch)
                self.writer.add_scalar('pgd_acc', pgd_acc_sum/num_batch, epoch)

            # save generator
            if epoch%20==0:
                E_file_name = models_path + self.model_name + 'E_epoch_' + str(epoch) + '.pth'
                advG_file_name = models_path + self.model_name + 'advG_epoch_' + str(epoch) + '.pth'
                defG_file_name = models_path + self.model_name + 'defG_epoch_' + str(epoch) + '.pth'
                recG_file_name = models_path + self.model_name + 'recG_epoch_' + str(epoch) + '.pth'
                torch.save(self.E.state_dict(), E_file_name)
                torch.save(self.advG.state_dict(), advG_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)
                torch.save(self.recG.state_dict(), recG_file_name)

        if self.writer:
            self.writer.close()

        # test performance
        self.test()

