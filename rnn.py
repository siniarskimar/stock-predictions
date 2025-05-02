import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn, optim, tensor

TorchDataset = torch.utils.data.Dataset
TorchDataLoader = torch.utils.data.DataLoader


DATASET_COLUMNS = [
    "Low",
    "Open",
    "Close",
    "Volume",
    "High",
    "Adjusted Close"
]


def verify_dataset(dataset: pd.DataFrame) -> (bool, str):
    REQUIRED_COLUMNS = set(DATASET_COLUMNS)

    if not REQUIRED_COLUMNS.issubset(dataset.columns):
        return (False, "dataset does not contain required columns")

    return (True, None)


class PredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=len(DATASET_COLUMNS),
                          hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, xs):
        out, _ = self.rnn(xs)
        return self.fc(out)


class PriceDataset(TorchDataset):
    def __init__(self,
                 dataset: pd.DataFrame,
                 device: torch.device,
                 window_size=5):
        valid_dataset, dataset_error = verify_dataset(dataset)
        if not valid_dataset:
            raise RuntimeError(dataset_error)

        self.close_column_idx = dataset.columns.get_loc("Close")
        self.wsize = window_size
        self.data = torch.tensor(
            dataset[DATASET_COLUMNS].to_numpy().astype(np.float32)
        ).to(device)

    def __len__(self):
        return len(self.data) - self.wsize - 1

    def __getitem__(self, idx):
        assert idx < len(self)

        if idx < 0:
            idx = len(self) + idx

        window = self.data[idx:idx+self.wsize+1]

        x = window[:-1]
        y = window[1:, self.close_column_idx]

        return x, y


def train_model(model: nn.Module,
                dataloader: TorchDataLoader,
                device: torch.device,
                optimizer: optim.Optimizer,
                loss_func: nn.Module,
                epochs=5000,
                epoch_callbacks=None):
    epoch_callbacks = epoch_callbacks or []

    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_func(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        for callback in epoch_callbacks:
            callback(epoch, loss.item())


def print_epoch_info(epoch: int, loss: float):
    print(f"epoch: {epoch} loss: {loss}")


def main():
    argparser = ArgumentParser('rnn.py')
    argparser.add_argument('dataset', type=Path)
    argparser.add_argument('-V', '--validation-dataset', type=Path)
    argparser.add_argument('-e', '--epochs', type=int, default=5000)

    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device.type}')

    if device.index is not None:
        print(f'Device index: {device.index}')

    model = PredictorModel()
    model.to(device)

    dataset = PriceDataset(pd.read_csv(
        args.dataset, usecols=DATASET_COLUMNS, sep=','), device)
    dataloader = TorchDataLoader(dataset, batch_size=1, shuffle=True)

    train_model(
        model,
        dataloader,
        device,
        optim.Adam(model.parameters(), lr=0.001),
        nn.MSELoss(),
        epochs=(args.epochs or None),
        epoch_callbacks=[print_epoch_info]
    )


if __name__ == "__main__":
    main()
