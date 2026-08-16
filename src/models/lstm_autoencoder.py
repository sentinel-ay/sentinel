import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output_layer(decoded)


def train_lstm_autoencoder(data, epochs=20, lr=1e-3):
    tensor = torch.tensor(data, dtype=torch.float32)
    model = LSTMAutoencoder(input_dim=tensor.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(tensor)
        loss = criterion(recon, tensor)
        loss.backward()
        optimizer.step()

    return model


def reconstruction_error(model, data):
    with torch.no_grad():
        recon = model(torch.tensor(data, dtype=torch.float32))
        err = torch.mean((recon - torch.tensor(data, dtype=torch.float32)) ** 2, dim=-1)
    return err.numpy()