import numpy as np

from src.models.lstm_autoencoder import LSTMAutoencoder, reconstruction_error, train_lstm_autoencoder


def detect_with_lstm(data: np.ndarray, threshold: float = 0.01):
    model = train_lstm_autoencoder(data, epochs=5)
    errors = reconstruction_error(model, data)
    return errors, errors > threshold