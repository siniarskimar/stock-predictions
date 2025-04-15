import sys

import torch
from torch import nn, optim, tensor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class PredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=6, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.to(torch.float32)
        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, xs):
        out, _ = self.rnn(xs)
        return self.fc(out)

    def fit(self, ts: tensor, ys: tensor) -> float:
        self.optimizer.zero_grad()
        y_pred = self.forward(ts)
        loss = self.loss(y_pred, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    EPOCHS = 5000

    model = PredictorModel()
    data = pd.read_csv(
        'NVDA.csv',
        sep=','
    )

    data = data.iloc[:, 1:]  # drop date
    data = data.to_numpy()
    data = data.astype(np.float32)

    T = torch.tensor(data[:-1])
    Y = torch.tensor(data[1:, 4]).unsqueeze(1)

    loss_over_epoch = []
    for epoch in range(EPOCHS):
        loss = model.fit(T, Y)
        loss_over_epoch.append(loss)
        sys.stdout.write("\r                                       \r")
        sys.stdout.write(f"epoch: {epoch} loss: {loss}")
        sys.stdout.flush()
    sys.stdout.write('\n')

    plt.plot(np.linspace(0, EPOCHS, EPOCHS), loss_over_epoch)
    plt.show()
