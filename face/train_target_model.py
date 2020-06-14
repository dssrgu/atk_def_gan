import torch
#import torchvision.datasets
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import load_models, load_dataset, num_correct

if __name__ == "__main__":
    use_cuda = True
    batch_size = 512

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    train_dataloader, test_dataloader = load_dataset('./dataset/real_and_fake_face', batch_size=batch_size, seed=0)

    # training the target model
    target_model, _ = load_models(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 60
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        num_corrects = 0
        total = 0
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            #print(logits_model, train_labels)
            #print(train_labels.cpu().numpy())
            #loss_model = F.binary_cross_entropy_with_logits(logits_model, train_labels.type_as(logits_model))
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()
            
            num_corrects += num_correct(logits_model, train_labels).item()
            total += train_labels.numel()
        

        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))
        print('accuracy in training set: %f\n' % (num_corrects/total))

    # save model
    targeted_model_file_name = './Face_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # MNIST test dataset
    # mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    # test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_corrects = 0
    total = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        logits_model = target_model(test_img)
        num_corrects += num_correct(logits_model, train_labels).item()
        total += train_labels.numel()

    print('accuracy in testing set: %f\n' % (num_corrects/total))
