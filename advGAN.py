import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
#import torchvision
import os
from test_adversarial_examples import test_full
from pgd_attack import PGD
from utils import weights_init


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 eps,
                 pgd_iter,
                 models_path,
                 out_path,
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
        self.pgd_iter = pgd_iter
        self.models_path = models_path
        self.out_path = out_path
        self.model_name = model_name
        self.writer = writer
        self.init_lr = init_lr
        self.rec_loss = nn.MSELoss()

        self.en_input_nc = image_nc
        self.E = models.Encoder(self.en_input_nc).to(device)
        self.defG = models.Generator(image_nc, adv=False).to(device)
        self.pgd = PGD(self.model, self.E, self.defG, self.device, self.eps)

        # initialize all weights
        self.E.apply(weights_init)
        self.defG.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                            lr=self.init_lr)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                               lr=self.init_lr)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(models_path + self.model_name):
            os.makedirs(models_path + self.model_name)

    # generate images for training
    def gen_images(self, x, labels):

        # make adv image
        adv_images = self.pgd.perturb(x, labels, itr=1)

        # make def(adv) image
        def_adv_images = self.defG(self.E(adv_images)) + adv_images
        def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

        # make def(nat) image
        def_images = self.defG(self.E(x)) + x
        def_images = torch.clamp(def_images, self.box_min, self.box_max)

        return adv_images, def_adv_images, def_images

    # performance tester
    def test(self):

        self.E.eval()
        self.defG.eval()

        test_full(self.device, self.model, self.E, self.defG, self.eps,
                  self.out_path, self.model_name, label_count=True, save_img=True)

        self.E.train()
        self.defG.train()

    # train single batch
    def train_batch(self, x, labels):

        # optimize E
        for i in range(1):

            # clear grad
            self.optimizer_E.zero_grad()
            self.optimizer_defG.zero_grad()

            _, def_adv_images, def_images = self.gen_images(x, labels)

            # def(adv) loss
            logits_def_adv = self.model(def_adv_images)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            # backprop
            loss = loss_def_adv + loss_def

            loss.backward()

            self.optimizer_E.step()
            self.optimizer_defG.step()

        # pgd performance check

        self.E.eval()
        self.defG.eval()

        pgd_acc_li = []

        for itr in self.pgd_iter:

            pgd_img = self.pgd.perturb(x, labels, itr=itr)

            for _ in range(itr):
                pgd_img = self.defG(self.E(pgd_img)) + pgd_img
                pgd_img = torch.clamp(pgd_img, self.box_min, self.box_max)

            pred = torch.argmax(self.model(pgd_img), 1)
            num_correct = torch.sum(pred == labels, 0)
            pgd_acc = num_correct.item()/len(labels)

            pgd_acc_li.append(pgd_acc)

        self.E.train()
        self.defG.train()

        return pgd_acc_li, torch.sum(loss).item(), \

    # main training function
    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.init_lr/10)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.init_lr/10)
            if epoch == 80:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.init_lr/100)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.init_lr/100)

            loss_sum = 0
            pgd_acc_li_sum = []

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                pgd_acc_li_batch, loss_batch = \
                    self.train_batch(images, labels)
                loss_sum += loss_batch
                pgd_acc_li_sum.append(pgd_acc_li_batch)

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_E: %.5f, loss_defG: %.5f" %
                  (epoch, loss_sum/num_batch,
                   loss_sum/num_batch))

            pgd_acc_li_sum = np.mean(np.array(pgd_acc_li_sum), axis=0)
            for idx in range(len(self.pgd_iter)):
                print("pgd iter %d acc.: %.5f" % (self.pgd_iter[idx], pgd_acc_li_sum[idx]))
            print()

            # write to tensorboard
            if self.writer:
                self.writer.add_scalar('loss', loss_sum/num_batch, epoch)
                for idx in range(len(self.pgd_iter)):
                    self.writer.add_scalar('pgd_acc_%d' % (self.pgd_iter[idx]), pgd_acc_li_sum[idx], epoch)

            # save generator
            if epoch%20==0:
                E_file_name = self.models_path + self.model_name + 'E_epoch_' + str(epoch) + '.pth'
                defG_file_name = self.models_path + self.model_name + 'defG_epoch_' + str(epoch) + '.pth'
                torch.save(self.E.state_dict(), E_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)

        if self.writer:
            self.writer.close()

        # test performance
        self.test()

