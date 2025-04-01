import torch
from torch import nn, tensor, optim

import pandas as pd


class PredictorModel(nn.Model):
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

    def fit(self, ts, ys):
        self.optimizer.zero_grad()
        y_pred = self.forward(ts)
        loss = self.loss(y_pred, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    model = PredictorModel()
