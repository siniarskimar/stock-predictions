import sys
from datetime import datetime

import torch
from torch import nn, optim

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class PredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=7, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.to(torch.float32)
        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, xs):
        out, _ = self.rnn(xs)
        return self.fc(out)

    def fit(self, ts, ys):
        self.optimizer.zero_grad()
        y_pred = self.forward(ts)
        loss = self.loss(y_pred, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    EPOCHS = 100

    model = PredictorModel()
    data = pd.read_csv(
        'NVDA.csv',
        sep=',').to_numpy()[1:]
    for row in data:
        row[0] = datetime.strptime(row[0], "%d-%m-%Y").timestamp()

    data = data.astype(np.float32)

    T = torch.from_numpy(data[:-1])
    Y = torch.from_numpy(data[1:, 5])
    loss_over_epoch = []
    for epoch in range(EPOCHS):
        loss = model.fit(T, Y)
        loss_over_epoch.append(loss)
        sys.stdout.write(f"\r                 \repoch: {epoch} loss: {loss}")
        sys.stdout.flush()
    sys.stdout.write('\n')

    print("prediction: " + model(torch.from_numpy(data[-1])))
    plt.plot(data[1:, 0], loss_over_epoch)
    plt.show()
