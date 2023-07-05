import torch
from torch import nn
import numpy as np


class FcNet(nn.Module):
    def __init__(self, db=20, depth=8, activation="tanh", dx=2, dy=1):
        super(FcNet, self).__init__()
        self.depth = depth
        self.db = db
        self.activation = activation
        fc = []
        for i in range(depth + 1):
            if i == 0:
                fc.append(nn.Linear(dx, db))
            elif i == depth:
                fc.append(nn.Linear(db, dy))
            else:
                fc.append(nn.Linear(db, db))
        self.fc = nn.ModuleList(fc)
        self.randominit()

    def activation_fn(self, x):
        # Sine Activation
        if self.activation == 'sin':
            x = torch.sin(x)
        # Sigmoid Variants
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'swish':
            x = x * torch.sigmoid(x)
        # Tanh Variants
        elif self.activation == 'tanh':
            x = torch.tanh(x)

        return x

    def forward(self, x):
        for i in range(self.depth):
            x = self.fc[i](x)
            x = self.activation_fn(x)
        return self.fc[self.depth](x)

    def randominit(self):
        for i in range(self.depth + 1):
            out_dim, in_dim = self.fc[i].weight.shape
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            self.fc[i].weight.data.normal_(0, xavier_stddev)
            self.fc[i].bias.data.fill_(0.0)


class PINN(FcNet):
    def __init__(
            self,
            pde,
            hidden_size=20,
            depth=8,
            activation="tanh",
    ):
        super().__init__(db=hidden_size, depth=depth, activation=activation, dx=len(pde.domain), dy=1)
        self.pde = pde
        self.params = nn.ModuleDict()
        self.randominit()

    def f(self, x):
        return self.pde.f(x, model=self, params=self.params)

    def loss(self, data, weights=None):
        loss = 0.
        losses = {}
        if weights is None:
            weights = {}
        for k in data.keys():
            if k in ['i', 'b', 'u']:
                x, u = data[k]
                loss_ = torch.mean((self.forward(x) - u) ** 2)
            elif k == 'f' or k == 'r':
                x, f = data[k]
                loss_ = torch.mean((self.f(x) - f) ** 2)
            else:
                raise RuntimeError('Unknown data type {}'.format(k))
            w = weights.get(k, 1.)
            losses[k] = loss_.item()
            loss += w * loss_
        return loss, losses

    def evaluate(self, data):
        x, u = data
        pred = self.forward(x)
        u_abs = torch.mean(torch.abs(u - pred))
        u_squared = torch.mean((u - pred) ** 2)
        u_rel_l2 = torch.norm(pred - u, p=2) / torch.norm(u, p=2)
        u_linf = torch.norm(pred - u, p=float('inf'))
        u_rel_linf = torch.norm(pred - u, p=float('inf')) / torch.norm(u, p=float('inf'))
        losses = {
            'u_abs': u_abs.item(),
            'u_squared': u_squared.item(),
            'u_rel_l2': u_rel_l2.item(),
            'u_linf': u_linf.item(),
            'u_rel_linf': u_rel_linf.item()
        }
        return losses

    def to_domain(self, x):
        l, u = np.array(self.pde.domain)[:, 0], np.array(self.pde.domain)[:, 1]
        return x * (u - l) + l



