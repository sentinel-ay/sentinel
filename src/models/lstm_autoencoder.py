import numpy as np
import torch
import torch.nn as nn


def make_windows(series, seq_len):
    """`(n, d)` sequence -> `(n - seq_len + 1, seq_len, d)` sliding windows."""
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    n = len(series) - seq_len + 1
    if n <= 0:
        raise ValueError(f"need at least seq_len={seq_len} samples, got {len(series)}")
    return np.stack([series[i:i + seq_len] for i in range(n)])


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output_layer(decoded)


def train_lstm_autoencoder(windows, hidden_dim, num_layers, epochs, lr, seed):
    """Train on normal-only windows of shape (n, seq_len, d)."""
    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(windows), dtype=torch.float32)
    model = LSTMAutoencoder(input_dim=x.shape[-1], hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        optimizer.step()

    return model


def reconstruction_error(model, windows):
    """Mean squared reconstruction error per window -> shape (n,)."""
    model.eval()
    x = torch.tensor(np.asarray(windows), dtype=torch.float32)
    with torch.no_grad():
        err = torch.mean((model(x) - x) ** 2, dim=(1, 2))
    return err.numpy()
