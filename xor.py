import torch
from torch import nn, tensor
from torch.optim import Adam

import matplotlib.pyplot as plt

import numpy as np


class XORModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            # nn.Sigmoid()

        )
        self.to(torch.float32)
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        # self.loss = nn.BCELoss()
        # self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X):
        return self.layers(X)

    def fit(self, X, Y):
        self.optimizer.zero_grad()
        y_predicted = self.forward(X)
        loss = self.loss(y_predicted, Y)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        loss.backward()
        self.optimizer.step()
        return loss.item()


Xs = tensor([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.float32)
Ys = tensor([[1], [1], [0], [0]], dtype=torch.float32)

xor_model = XORModel()

EPOCHS = 7000

loss_over_epochs = [xor_model.fit(Xs, Ys) for i in range(EPOCHS)]

print(loss_over_epochs[-1])
plt.plot(np.linspace(0, EPOCHS, EPOCHS), loss_over_epochs)
plt.show()
