import sys
import argparse
import torch
from torch import nn, optim, tensor

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


class ImprovedPredictorModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Używamy LSTM zamiast prostego RNN
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dodanie warstw liniowych z aktywacją
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
        # Przejście na float32
        self.to(torch.float32)
        
        # Zoptymalizowany optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )

    def forward(self, xs):
        # LSTM zwraca (output, (h_n, c_n))
        lstm_out, _ = self.lstm(xs)
        
        # Bierzemy tylko ostatnią wartość z sekwencji dla predykcji
        lstm_out = lstm_out[:, -1, :]
        
        # Przetworzenie przez warstwy liniowe z aktywacją
        x = self.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x

    def fit(self, ts: tensor, ys: tensor) -> float:
        self.optimizer.zero_grad()
        y_pred = self.forward(ts)
        loss = self.loss(y_pred, ys)
        loss.backward()
        
        # Clipping gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ulepszony model LSTM dla predykcji cen akcji')
    parser.add_argument('--train', type=str, required=True, help='Ścieżka do pliku CSV z danymi treningowymi')
    parser.add_argument('--test', type=str, required=True, help='Ścieżka do pliku CSV z danymi testowymi')
    parser.add_argument('--epochs', type=int, default=2000, help='Liczba epok trenowania')
    parser.add_argument('--target-column', type=str, default='Close', help='Kolumna do przewidywania')
    parser.add_argument('--batch-size', type=int, default=64, help='Rozmiar batcha (0 dla całego zbioru)')
    parser.add_argument('--sequence-length', type=int, default=10, help='Długość sekwencji wejściowej')
    parser.add_argument('--hidden-size', type=int, default=64, help='Rozmiar warstwy ukrytej LSTM')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Współczynnik uczenia')
    return parser.parse_args()


def load_data(filepath):
    data = pd.read_csv(filepath, sep=',')
    # Konwersja kolumny daty na format datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    return data


def prepare_sequences(data, seq_length, target_idx):
    """Przygotowuje sekwencje danych do trenowania modelu LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Sekwencja wejściowa
        seq = data[i:i+seq_length]
        X.append(seq)
        
        # Wartość docelowa (następna wartość po sekwencji)
        target = data[i+seq_length, target_idx]
        y.append(target)
        
    return np.array(X), np.array(y).reshape(-1, 1)


def create_data_loaders(X, y, batch_size):
    """Tworzy data loadery dla PyTorch z sekwencji danych"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Tworzymy TensorDataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Jeśli batch_size to 0, używamy całego zbioru
    actual_batch_size = len(dataset) if batch_size == 0 else batch_size
    
    # Tworzymy DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=actual_batch_size, shuffle=True
    )
    
    return data_loader


def train_model_with_batches(model, data_loader, epochs, device, patience=100):
    """Trenuje model z użyciem mini-batchy i early stopping"""
    model.train()
    
    loss_history = []
    best_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for X_batch, y_batch in data_loader:
            # Przeniesienie danych na odpowiednie urządzenie
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Trenowanie modelu
            loss = model.fit(X_batch, y_batch)
            epoch_loss += loss
            batch_count += 1
        
        # Średnia strata na epokę
        avg_epoch_loss = epoch_loss / batch_count
        loss_history.append(avg_epoch_loss)
        
        # Aktualizacja learning rate schedulera
        model.scheduler.step(avg_epoch_loss)
        
        # Wyświetlenie postępu
        sys.stdout.write("\r                                       \r")
        sys.stdout.write(f"epoka: {epoch+1}/{epochs} strata: {avg_epoch_loss:.8f}")
        sys.stdout.flush()
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"\nEarly stopping po {epoch+1} epokach, brak poprawy przez {patience} epok.")
                break
    
    sys.stdout.write('\n')
    return loss_history


def predict_sequences(model, data, scalers, target_idx, seq_length, device):
    """Przewiduje sekwencje danych używając trenowanego modelu"""
    model.eval()
    predictions = []
    feature_scaler, _ = scalers
    
    with torch.no_grad():
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            
            # Predykcja
            pred = model(seq_tensor).cpu().numpy()[0, 0]
            
            # Odwrócenie skalowania
            if feature_scaler:
                # Przygotowanie do odwrócenia skalowania (tworzymy pełny wektor)
                dummy = np.zeros((1, data.shape[1]))
                dummy[0, target_idx] = pred
                # Odwrócenie skalowania tylko dla docelowej kolumny
                pred = feature_scaler.inverse_transform(dummy)[0, target_idx]
            
            predictions.append(pred)
    
    return np.array(predictions)


def prepare_data_for_training(train_data, test_data, target_column, seq_length=10, normalize=True):
    """Przygotowuje dane do trenowania, włącznie z normalizacją"""
    # Zapisanie dat do wizualizacji
    train_dates = train_data['Date'].copy() if 'Date' in train_data.columns else None
    test_dates = test_data['Date'].copy() if 'Date' in test_data.columns else None
    
    # Usunięcie kolumny daty
    if 'Date' in train_data.columns:
        train_data = train_data.drop('Date', axis=1)
    if 'Date' in test_data.columns:
        test_data = test_data.drop('Date', axis=1)
    
    # Indeks kolumny docelowej
    if target_column not in train_data.columns:
        raise ValueError(f"Kolumna '{target_column}' nie istnieje w danych")
    
    target_idx = train_data.columns.get_loc(target_column)
    
    # Normalizacja danych
    feature_scaler = None
    target_scaler = None
    if normalize:
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        train_array = feature_scaler.fit_transform(train_data.values)
        test_array = feature_scaler.transform(test_data.values)
    else:
        train_array = train_data.values.astype(np.float32)
        test_array = test_data.values.astype(np.float32)
    
    # Przygotowanie sekwencji dla treningu
    X_train, y_train = prepare_sequences(train_array, seq_length, target_idx)
    X_test, y_test = prepare_sequences(test_array, seq_length, target_idx)
    
    # Data dla generowania predykcji
    train_seq_dates = None
    test_seq_dates = None
    
    if train_dates is not None:
        train_seq_dates = train_dates[seq_length:].reset_index(drop=True)
    if test_dates is not None:
        test_seq_dates = test_dates[seq_length:].reset_index(drop=True)
    
    return (
        (X_train, y_train), (X_test, y_test),
        (train_array, test_array),
        (train_seq_dates, test_seq_dates),
        target_idx,
        (feature_scaler, target_scaler)
    )


def plot_results(train_dates, train_actual, train_pred, 
                 test_dates, test_actual, test_pred, 
                 target_column, loss_history):
    """Wizualizacja wyników predykcji i historii straty"""
    plt.figure(figsize=(15, 10))
    
    # Wykres 1: Rzeczywiste vs Przewidywane wartości
    plt.subplot(2, 1, 1)
    plt.title(f'Rzeczywiste vs Przewidywane wartości ({target_column})')
    
    # Dane treningowe
    if train_dates is not None:
        plt.plot(train_dates, train_actual, 'b-', label='Dane treningowe')
        plt.plot(train_dates, train_pred, 'r--', label='Przewidywania (trening)')
    else:
        plt.plot(range(len(train_actual)), train_actual, 'b-', label='Dane treningowe')
        plt.plot(range(len(train_pred)), train_pred, 'r--', label='Przewidywania (trening)')
    
    # Dane testowe
    if test_dates is not None:
        plt.plot(test_dates, test_actual, 'g-', label='Dane testowe')
        plt.plot(test_dates, test_pred, 'm--', label='Przewidywania (test)')
    else:
        offset = len(train_actual)
        plt.plot(range(offset, offset + len(test_actual)), test_actual, 'g-', label='Dane testowe')
        plt.plot(range(offset, offset + len(test_pred)), test_pred, 'm--', label='Przewidywania (test)')
    
    plt.xlabel('Data')
    plt.ylabel('Wartość')
    plt.legend()
    
    # Formatowanie dat na osi X
    if train_dates is not None:
        plt.gcf().autofmt_xdate()
        date_formatter = DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_formatter)
    
    # Wykres 2: Funkcja straty
    plt.subplot(2, 1, 2)
    plt.title('Funkcja straty w czasie')
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel('Epoka')
    plt.ylabel('Strata (MSE)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def calculate_metrics(actual, predicted):
    """Oblicza metryki oceny modelu"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def main():
    args = parse_arguments()
    
    # Określenie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Używane urządzenie: {device.type}')
    if device.type == 'cuda' and torch.cuda.get_device_properties(0) is not None:
        print(f'Model GPU: {torch.cuda.get_device_name(0)}')
    
    # Wczytanie danych
    train_data = load_data(args.train)
    test_data = load_data(args.test)
    print(f'Wczytano dane treningowe: {train_data.shape}')
    print(f'Wczytano dane testowe: {test_data.shape}')
    
    # Sprawdzenie czy kolumna docelowa istnieje
    if args.target_column not in train_data.columns:
        print(f"Kolumna '{args.target_column}' nie istnieje w danych. Dostępne kolumny: {train_data.columns.tolist()}")
        return
    
    # Parametry
    seq_length = args.sequence_length
    
    # Przygotowanie danych
    print(f'Przygotowywanie danych z długością sekwencji {seq_length}...')
    (X_train, y_train), (X_test, y_test), (train_array, test_array), (train_dates, test_dates), target_idx, scalers = prepare_data_for_training(
        train_data, test_data, args.target_column, seq_length
    )
    
    # Tworzenie data loaderów
    train_loader = create_data_loaders(X_train, y_train, args.batch_size)
    
    # Inicjalizacja modelu
    input_size = X_train.shape[2]  # Liczba cech wejściowych
    model = ImprovedPredictorModel(input_size=input_size, hidden_size=args.hidden_size)
    model.to(device)
    
    # Zmiana learning rate jeśli podano w argumentach
    if args.learning_rate != 0.001:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    
    # Wyświetlenie architektury modelu
    print(f'\nArchitektura modelu:')
    print(model)
    print(f'\nParametry modelu: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Trenowanie modelu
    print(f'\nRozpoczęcie trenowania modelu ({args.epochs} epok)...')
    loss_history = train_model_with_batches(model, train_loader, args.epochs, device)
    
    # Generowanie predykcji
    print('Generowanie prognoz...')
    train_pred = predict_sequences(model, train_array, scalers, target_idx, seq_length, device)
    test_pred = predict_sequences(model, test_array, scalers, target_idx, seq_length, device)
    
    # Przygotowanie rzeczywistych wartości do porównania
    train_actual = train_array[seq_length:, target_idx]
    test_actual = test_array[seq_length:, target_idx]
    
    # Jeśli dane są znormalizowane, cofamy normalizację
    feature_scaler, _ = scalers
    if feature_scaler:
        # Przygotowanie pełnych wektorów do odwrócenia normalizacji
        train_dummy = np.zeros((len(train_actual), train_array.shape[1]))
        train_dummy[:, target_idx] = train_actual
        train_actual = feature_scaler.inverse_transform(train_dummy)[:, target_idx]
        
        test_dummy = np.zeros((len(test_actual), test_array.shape[1]))
        test_dummy[:, target_idx] = test_actual
        test_actual = feature_scaler.inverse_transform(test_dummy)[:, target_idx]
    
    # Obliczanie metryk
    train_metrics = calculate_metrics(train_actual, train_pred)
    test_metrics = calculate_metrics(test_actual, test_pred)
    
    print('\nMetryki dla danych treningowych:')
    for metric, value in train_metrics.items():
        print(f'  {metric}: {value:.6f}')
    
    print('\nMetryki dla danych testowych:')
    for metric, value in test_metrics.items():
        print(f'  {metric}: {value:.6f}')
    
    # Wizualizacja wyników
    plot_results(train_dates, train_actual, train_pred, 
                test_dates, test_actual, test_pred, 
                args.target_column, loss_history)


if __name__ == "__main__":
    main()