import torch
import numpy as np
import models
import torch.nn.functional as F
#import torchvision
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
                 E_lr,
                 defG_lr):
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
        self.E_lr = E_lr
        self.defG_lr = defG_lr

        self.en_input_nc = image_nc
        self.E = models.Encoder(image_nc).to(device)
        self.defG = models.Generator(adv=False).to(device)
        self.pgd = PGD(self.model, self.E, self.defG, self.device, self.eps)

        # initialize all weights
        self.E.apply(weights_init)
        self.defG.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                            lr=self.E_lr)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                               lr=self.defG_lr)

    # generate images for training
    def gen_images(self, x, labels):

        # pgd image
        pgd_images = self.pgd.perturb(x, labels, itr=1)

        # def(pgd) image
        def_pgd_images = self.defG(self.E(pgd_images)) + pgd_images
        def_pgd_images = torch.clamp(def_pgd_images, self.box_min, self.box_max)

        # make def(nat) image
        def_images = self.defG(self.E(x)) + x
        def_images = torch.clamp(def_images, self.box_min, self.box_max)

        return pgd_images, def_pgd_images, def_images

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

        # optimize E, def
        for i in range(1):

            # clear grad
            self.optimizer_E.zero_grad()
            self.optimizer_defG.zero_grad()

            pgd_images, def_pgd_images, def_images = self.gen_images(x, labels)

            # def(pgd) loss
            logits_def_pgd = self.model(def_pgd_images)
            loss_def_pgd = F.cross_entropy(logits_def_pgd, labels)

            # def(nat) loss
            logits_def = self.model(def_images)
            loss_def = F.cross_entropy(logits_def, labels)

            # backprop
            loss = loss_def + loss_def_pgd

            loss.backward()

            self.optimizer_E.step()
            self.optimizer_defG.step()

        # pgd performance check

        self.E.eval()
        self.defG.eval()

        pgd_acc_li = []
        pgd_nat_acc_li = []

        pgd_nat_images = self.pgd.perturb(x, labels, itr=0)

        # obsufcated check
        pgd_nat_images = self.defG(self.E(pgd_nat_images)) + pgd_nat_images
        pgd_nat_images = torch.clamp(pgd_nat_images, self.box_min, self.box_max)

        pred = torch.argmax(self.model(pgd_images), 1)
        num_correct = torch.sum(pred == labels, 0)
        pgd_acc = num_correct.item()/len(labels)

        pgd_acc_li.append(pgd_acc)

        pred = torch.argmax(self.model(pgd_nat_images), 1)
        num_correct = torch.sum(pred == labels, 0)
        pgd_nat_acc = num_correct.item()/len(labels)

        pgd_nat_acc_li.append(pgd_nat_acc)

        self.E.train()
        self.defG.train()

        return pgd_acc_li, pgd_nat_acc_li, torch.sum(loss).item()

    # main training function
    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.E_lr/10)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.defG_lr/10)
            if epoch == 80:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.E_lr/100)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.defG_lr/100)

            loss_sum = 0
            pgd_acc_li_sum = []
            pgd_nat_acc_li_sum = []

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                pgd_acc_li_batch, pgd_nat_acc_li_batch, loss_batch = \
                    self.train_batch(images, labels)
                loss_sum += loss_batch
                pgd_acc_li_sum.append(pgd_acc_li_batch)
                pgd_nat_acc_li_sum.append(pgd_nat_acc_li_batch)

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_E: %.5f" %
                  (epoch, loss_sum/num_batch))

            pgd_acc_li_sum = np.mean(np.array(pgd_acc_li_sum), axis=0)
            for idx in range(len(self.pgd_iter)):
                print("pgd iter %d acc.: %.5f" % (self.pgd_iter[idx], pgd_acc_li_sum[idx]))
            
            pgd_nat_acc_li_sum = np.mean(np.array(pgd_nat_acc_li_sum), axis=0)
            for idx in range(len(self.pgd_iter)):
                print("pgd nat iter %d acc.: %.5f" % (self.pgd_iter[idx], pgd_nat_acc_li_sum[idx]))
            print()

            # write to tensorboard
            if self.writer:
                self.writer.add_scalar('loss', loss_sum/num_batch, epoch)
                for idx in range(len(self.pgd_iter)):
                    self.writer.add_scalar('pgd_acc_%d' % (self.pgd_iter[idx]), pgd_acc_li_sum[idx], epoch)
                    self.writer.add_scalar('pgd_nat_acc_%d' % (self.pgd_iter[idx]), pgd_nat_acc_li_sum[idx], epoch)

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

