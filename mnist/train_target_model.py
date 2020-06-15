import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import  MNIST_target_net

from pgd_attack import PGD
import argparse
from utils import num_correct

if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data', default='./dataset/real_and_fake_face')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epsilon', default=0.3, type=float)
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # training the target model
    target_model = MNIST_target_net().to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = args.epoch

    pgd = PGD(target_model, None, None, device, args.epsilon, 7, args.epsilon/4)
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)

        num_corrects = 0
        adv_num_corrects = 0
        total = 0
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            adv_imgs = pgd.perturb(train_imgs, train_labels, itr=0)
            adv_logits_model = target_model(adv_imgs)
            loss_model = F.cross_entropy(adv_logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

            num_corrects += num_correct(logits_model, train_labels).item()
            adv_num_corrects += num_correct(adv_logits_model, train_labels).item()
            total += train_labels.numel()


        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))
        print('accuracy in training set: %f' % (num_corrects/total))
        print('adv accuracy in training set: %f' % (adv_num_corrects/total))

    # save model
    targeted_model_file_name = './MNIST_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # MNIST test dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_corrects = 0
    adv_num_corrects = 0
    total = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        logits_model = target_model(test_img)
        adv_img = pgd.perturb(test_img, test_label, itr=0)
        adv_logits_model = target_model(adv_img)

        num_corrects += num_correct(logits_model, test_label).item()
        adv_num_corrects += num_correct(adv_logits_model, test_label).item()
        total += test_label.numel()

    print('accuracy in testing set: %f\n' % (num_corrects/total))
    print('accuracy in testing set: %f\n' % (adv_num_corrects/total))
