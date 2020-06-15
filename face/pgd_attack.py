import numpy as np
import torch
import torch.nn as nn
from wideresnet import WideResNet
from utils import normalized_eval

class PGD(object):
    def __init__(self, model=None, enc=None, defG=None, device=None, eps=0.03125, num_steps=10, step_size=0.0078125):
        self.model = model
        self.enc = enc
        self.defG = defG
        self.device = device
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_ori, y, itr=0):

        X_nat = np.copy(X_ori.cpu())
        X = np.copy(X_nat)

        for i in range(self.num_steps):
            X_var = torch.from_numpy(X).to(self.device)
            X_var.requires_grad = True

            if itr > 0:
                X_def = self.defG(self.enc(X_var)) + X_var
                X_def = torch.clamp(X_def, 0, 1)

                for _ in range(itr-1):
                    X_def = self.defG(self.enc(X_def)) + X_def
                    X_def = torch.clamp(X_def, 0, 1)
                
                scores = normalized_eval(X_def, self.model)

            else:
                scores = normalized_eval(X_var, self.model)

            loss = self.loss_fn(scores, y)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.step_size * np.sign(grad)

            X = np.clip(X, X_nat - self.eps, X_nat + self.eps)
            X = np.clip(X, 0, 1)

        return torch.from_numpy(X).to(self.device)

if __name__ == '__main__':
    
    use_cuda = True
    batch_size = 128
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./model_best.pth.tar"
    target_model = WideResNet().to(device)
    checkpoint = torch.load(pretrained_model, map_location=device)
    target_model.load_state_dict(checkpoint)

    pgd = PGD(target_model, device=device)
    
