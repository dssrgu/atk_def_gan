import numpy as np
import torch
import torch.nn as nn

class PGD(object):
    def __init__(self, model=None, enc=None, defG=None, device=None, eps=0.3, num_steps=10, step_size=0.075):
        self.model = model
        self.enc = enc
        self.defG = defG
        self.device = device
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_ori, y, it=0):

        X_nat = np.copy(X_ori)
        X = np.copy(X_nat)

        for i in range(self.num_steps):
            X_var = torch.from_numpy(X).to(self.device)
            X_var.requires_grad = True

            if it > 0:
                X_def = self.defG(self.enc(X_var)) + X_var
                X_def = torch.clamp(X_def, 0, 1)

                for _ in range(it-1):
                    X_def = self.defG(self.enc(X_def)) + X_def
                    X_def = torch.clamp(X_def, 0, 1)
                
                scores = self.model(X_def)

            else:
                scores = self.model(X_var)

            loss = self.loss_fn(scores, y)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.step_size * np.sign(grad)

            X = np.clip(X, X_nat - self.eps, X_nat + self.eps)
            X = np.clip(X, 0, 1)

        return torch.from_numpy(X).to(self.device)
