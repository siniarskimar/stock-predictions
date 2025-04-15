import torch
from torch import nn, tensor
from torch.optim import Adam

import matplotlib.pyplot as plt

import numpy as np


class SineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True)
        self.fc = nn.Linear(2, 1)

        self.to(torch.float32)
        self.optimizer = Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, xs):
        out, _ = self.rnn(xs)
        return self.fc(out)

    def fit(self, xs, ys):
        self.optimizer.zero_grad()
        ys_pred = self.forward(xs)
        loss = self.loss(ys_pred, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    model = SineModel()
    time_steps = np.linspace(0, 100, 100, dtype=np.float32)
    data = np.sin(time_steps)

    X = tensor(data[:-1]).reshape(1, -1, 1)
    Y = tensor(data[1:]).reshape(1, -1, 1)

    EPOCHS = 1000
    loss_over_epoch = [model.fit(X, Y) for _ in range(EPOCHS)]

    print(loss_over_epoch[-1])
    plt.plot(np.linspace(0, EPOCHS, EPOCHS), loss_over_epoch)
    plt.show()
