import sys
import torch
from torch import nn, optim, tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

minArr = np.full(shape=6, fill_value=10**8, dtype=float)
maxArr = np.full(shape=6, fill_value=0, dtype=float)


def min_max(data: np.array) -> None:

    for row in data:
        for idx in range(len(row)):
            if (row[idx] > maxArr[idx]):
                maxArr[idx] = row[idx]
            if (row[idx] < minArr[idx]):
                minArr[idx] = row[idx]

    for idx, sequence in enumerate(data):
        for i, el in enumerate(sequence):
            el = (el - minArr[i])/(maxArr[i] - minArr[i])
            data[idx][i] = el


def create_sets(data: np.array, length: int) -> np.array:
    x, y = [], []
    for i in range(len(data) - length):
        x.append(data[i:i + length])
        y.append(data[i + length])
    return np.array(x), np.array(y)


class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.rnn = nn.RNN(input_size=6, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 6)
        self.to(torch.float32)

    def forward(self, xs: tensor) -> tensor:
        out, _ = self.rnn(xs)
        # print("_", _)
        # print("out", out)
        # print(out.shape)
        # print("out", out[:, -1, :])
        # print("out: ", self.fc(out[:, -1, :]))
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    EPOCHS = 200

    model = PredictorModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    data = pd.read_csv(
        'NVDA.csv',
        sep=','
    )

    data = data.iloc[:, 1:]  # drop date
    data = data.to_numpy()
    data = data.astype(np.float32)

    min_max(data)

    x, y = create_sets(data, 10)
    # print(x)
    # exit(420)

    split_index = int(len(data) * 0.85)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()
        y_pred = model.forward(x_train)

        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        # print(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            test_loss = loss_fn(test_pred, y_test)
            print(f"Test Loss:{test_loss.item()}")

    print(y_test)
    print(" ")
    print(test_pred)

    # plt.plot(test_pred[:len(y_test), 2], label="predicted values")
    # plt.plot(y_test[:, 2], label="true values")
    # plt.show()

    plt.plot(test_pred, label="predicted")
    plt.plot(y_test, label="true values")
    plt.show()
    plt.show()
