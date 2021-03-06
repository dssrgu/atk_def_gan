import torch
import numpy as np
import models
import torch.nn.functional as F
# import torchvision
from test_adversarial_examples import test_full
from pgd_attack import PGD
from utils import normalized_eval
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
                 advG_lr,
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
        self.advG_lr = advG_lr
        self.defG_lr = defG_lr

        self.en_input_nc = image_nc
        self.E = models.Encoder(image_nc).to(device)
        self.defG = models.Generator(adv=False).to(device)
        self.advG = models.Generator(y_dim=model_num_labels, adv=True).to(device)
        self.pgd = PGD(self.model, self.E, self.defG, self.device, self.eps)

        # initialize all weights
        self.E.apply(weights_init)
        self.defG.apply(weights_init)
        self.advG.apply(weights_init)

        # initialize optimizers
        self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                            lr=self.E_lr)
        self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                               lr=self.defG_lr)
        self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                               lr=self.advG_lr)

    # generate images for training
    def gen_images(self, x, labels, adv=True, def_adv=True, def_nat=True):

        results = []

        # random target labels
        target_labels = torch.randint_like(labels, 0, self.model_num_labels)
        target_one_hot = torch.eye(self.model_num_labels, device=self.device)[target_labels]
        target_one_hot = target_one_hot.view(-1, self.model_num_labels, 1, 1)

        x_encoded = self.E(x)

        if adv or def_adv:
            # make adv image
            adv_images = self.advG(x_encoded, target_one_hot) * self.eps + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            results.append(adv_images)

        if def_adv:
            # make def(adv) image
            def_adv_images = self.defG(self.E(adv_images)) + adv_images
            def_adv_images = torch.clamp(def_adv_images, self.box_min, self.box_max)

            results.append(def_adv_images)

        if def_nat:
            # make def(nat) image
            def_images = self.defG(x_encoded) + x
            def_images = torch.clamp(def_images, self.box_min, self.box_max)

            results.append(def_images)

        results.append(target_labels)

        return results

    # performance tester
    def test(self):

        self.E.eval()
        self.defG.eval()

        test_full(self.device, self.model, self.E, self.defG, self.advG, self.eps,
                  self.out_path, self.model_name, label_count=True, save_img=True)

        self.E.train()
        self.defG.train()

    # train single batch
    def train_batch(self, x, labels, batch_num):

        # optimize E
        for i in range(1):
            # clear grad
            self.optimizer_E.zero_grad()

            adv_images, def_adv_images, def_images, target_labels = self.gen_images(x, labels)

            # adv loss
            logits_adv = normalized_eval(adv_images, self.model)
            loss_adv = F.cross_entropy(logits_adv, target_labels)

            # def(adv) loss
            logits_def_adv = normalized_eval(def_adv_images, self.model)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def(nat) loss
            logits_def = normalized_eval(def_images, self.model)
            loss_def = F.cross_entropy(logits_def, labels)

            # backprop
            loss_E = loss_adv + loss_def_adv + loss_def

            loss_E.backward()

            self.optimizer_E.step()

        # optimize G
        for i in range(1):
            # clear grad
            self.optimizer_advG.zero_grad()
            adv_images, def_adv_images, target_labels = self.gen_images(x, labels, def_nat=False)

            # adv loss
            logits_adv = normalized_eval(adv_images, self.model)
            loss_adv = F.cross_entropy(logits_adv, target_labels)

            # def(adv) loss
            logits_def_adv = normalized_eval(def_adv_images, self.model)
            loss_def_adv = F.cross_entropy(logits_def_adv, target_labels)

            # backprop
            loss_advG = loss_adv + loss_def_adv

            loss_advG.backward()

            self.optimizer_advG.step()

        # optimize defG
        for i in range(1):
            # clear grad
            self.optimizer_defG.zero_grad()

            _, def_adv_images, def_images, _ = self.gen_images(x, labels, adv=False)

            # def(adv) loss
            logits_def_adv = normalized_eval(def_adv_images, self.model)
            loss_def_adv = F.cross_entropy(logits_def_adv, labels)

            # def loss
            logits_def = normalized_eval(def_images, self.model)
            loss_def = F.cross_entropy(logits_def, labels)

            # backprop
            loss_defG = loss_def_adv + loss_def

            loss_defG.backward()

            self.optimizer_defG.step()

        if batch_num == 0:

            self.E.eval()
            self.advG.eval()
            self.defG.eval()

            # adv, def performance check
            adv_pred = torch.argmax(normalized_eval(adv_images, self.model), 1)
            adv_correct = torch.sum(adv_pred == labels, 0)
            adv_acc = adv_correct.item() / len(labels)

            def_adv_pred = torch.argmax(normalized_eval(def_adv_images, self.model), 1)
            def_adv_correct = torch.sum(def_adv_pred == labels, 0)
            def_adv_acc = def_adv_correct.item() / len(labels)

            def_pred = torch.argmax(normalized_eval(def_images, self.model), 1)
            def_correct = torch.sum(def_pred == labels, 0)
            def_acc = def_correct.item() / len(labels)

            nat_pred = torch.argmax(normalized_eval(x, self.model), 1)
            nat_correct = torch.sum(nat_pred == labels, 0)
            nat_acc = nat_correct.item() / len(labels)

            print('adv mean perturbation: %.5f' % torch.abs(adv_images - x).mean().item())
            print('def_adv mean perturbation: %.5f' % torch.abs(def_adv_images - x).mean().item())
            print('def mean perturbation: %.5f' % torch.abs(def_images - x).mean().item())
            print()

            # pgd performance check

            pgd_acc_li = []
            pgd_nat_acc_li = []
            for itr in self.pgd_iter:

                pgd_img = self.pgd.perturb(x, labels, itr=itr)
                pgd_nat_img = self.pgd.perturb(x, labels, itr=0)

                for _ in range(itr):
                    pgd_img = self.defG(self.E(pgd_img)) + pgd_img
                    pgd_img = torch.clamp(pgd_img, self.box_min, self.box_max)

                    # obfuscated check
                    pgd_nat_img = self.defG(self.E(pgd_nat_img)) + pgd_nat_img
                    pgd_nat_img = torch.clamp(pgd_nat_img, self.box_min, self.box_max)

                pred = torch.argmax(normalized_eval(pgd_img, self.model), 1)
                num_correct = torch.sum(pred == labels, 0)
                pgd_acc = num_correct.item() / len(labels)

                pgd_acc_li.append(pgd_acc)

                pred = torch.argmax(normalized_eval(pgd_nat_img, self.model), 1)
                num_correct = torch.sum(pred == labels, 0)
                pgd_nat_acc = num_correct.item() / len(labels)

                pgd_nat_acc_li.append(pgd_nat_acc)

            self.defG.train()
            self.advG.train()
            self.E.train()

        else:
            pgd_acc_li = None
            pgd_nat_acc_li = None

            adv_acc = None
            def_adv_acc = None
            def_acc = None
            nat_acc = None

        return pgd_acc_li, pgd_nat_acc_li, torch.sum(loss_E).item(), torch.sum(loss_advG).item(), \
               torch.sum(loss_defG).item(), \
               adv_acc, def_adv_acc, def_acc, nat_acc

    # main training function
    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs + 1):

            if epoch == 50:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.E_lr / 10)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.defG_lr / 10)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                       lr=self.advG_lr / 10)
            if epoch == 80:
                self.optimizer_E = torch.optim.Adam(self.E.parameters(),
                                                    lr=self.E_lr / 100)
                self.optimizer_defG = torch.optim.Adam(self.defG.parameters(),
                                                       lr=self.defG_lr / 100)
                self.optimizer_advG = torch.optim.Adam(self.advG.parameters(),
                                                       lr=self.advG_lr / 100)

            loss_E_sum = 0
            loss_defG_sum = 0
            loss_advG_sum = 0
            pgd_acc_li_sum = []
            pgd_nat_acc_li_sum = []
            nat_acc_sum = 0
            adv_acc_sum = 0
            def_adv_acc_sum = 0
            def_acc_sum = 0

            for i, data in enumerate(train_dataloader, start=0):

                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                pgd_acc_li_batch, pgd_nat_acc_li_batch, loss_E_batch, loss_advG_batch, loss_defG_batch, \
                adv_acc, def_adv_acc, def_acc, nat_acc = \
                    self.train_batch(images, labels, i)

                loss_E_sum += loss_E_batch
                loss_advG_sum += loss_advG_batch
                loss_defG_sum += loss_defG_batch

                if pgd_acc_li_batch:
                    pgd_acc_li_sum.append(pgd_acc_li_batch)
                    pgd_nat_acc_li_sum.append(pgd_nat_acc_li_batch)
                    nat_acc_sum += nat_acc
                    adv_acc_sum += adv_acc
                    def_adv_acc_sum += def_adv_acc
                    def_acc_sum += def_acc

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_E: %.5f, loss_advG: %.5f, loss_defG: %.5f" %
                  (epoch, loss_E_sum / num_batch, loss_advG_sum / num_batch,
                   loss_defG_sum / num_batch))

            pgd_acc_li_sum = np.mean(np.array(pgd_acc_li_sum), axis=0)
            for idx in range(len(self.pgd_iter)):
                print("pgd iter %d acc.: %.5f" % (self.pgd_iter[idx], pgd_acc_li_sum[idx]))

            pgd_nat_acc_li_sum = np.mean(np.array(pgd_nat_acc_li_sum), axis=0)
            for idx in range(len(self.pgd_iter)):
                print("pgd nat iter %d acc.: %.5f" % (self.pgd_iter[idx], pgd_nat_acc_li_sum[idx]))

            print("nat acc.: %.5f" % (nat_acc_sum))
            print("adv acc.: %.5f" % (adv_acc_sum))
            print("def_adv acc.: %.5f" % (def_adv_acc_sum))
            print("def acc.: %.5f" % (def_acc_sum))

            print()

            # write to tensorboard
            if self.writer:
                self.writer.add_scalar('loss_E', loss_E_sum / num_batch, epoch)
                self.writer.add_scalar('loss_advG', loss_advG_sum / num_batch, epoch)
                self.writer.add_scalar('loss_defG', loss_defG_sum / num_batch, epoch)

                for idx in range(len(self.pgd_iter)):
                    self.writer.add_scalar('pgd_acc_%d' % (self.pgd_iter[idx]), pgd_acc_li_sum[idx], epoch)
                    self.writer.add_scalar('pgd_nat_acc_%d' % (self.pgd_iter[idx]), pgd_nat_acc_li_sum[idx], epoch)

                self.writer.add_scalar('nat_acc', nat_acc_sum, epoch)
                self.writer.add_scalar('adv_acc', adv_acc_sum, epoch)
                self.writer.add_scalar('def_adv_acc', def_adv_acc_sum, epoch)
                self.writer.add_scalar('def_acc', def_acc_sum, epoch)

            # save generator
            if epoch % 20 == 0:
                E_file_name = self.models_path + self.model_name + 'E_epoch_' + str(epoch) + '.pth'
                advG_file_name = self.models_path + self.model_name + 'advG_epoch_' + str(epoch) + '.pth'
                defG_file_name = self.models_path + self.model_name + 'defG_epoch_' + str(epoch) + '.pth'
                torch.save(self.E.state_dict(), E_file_name)
                torch.save(self.advG.state_dict(), advG_file_name)
                torch.save(self.defG.state_dict(), defG_file_name)

        if self.writer:
            self.writer.close()

        # test performance
        self.test()
