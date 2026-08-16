import numpy as np

from src.models.lstm_autoencoder import make_windows, reconstruction_error, train_lstm_autoencoder


def _scores_to_timesteps(window_errors, n, seq_len):
    """A window ending at index i scores timestep i; pad the first seq_len-1 steps."""
    scores = np.empty(n, dtype=float)
    scores[seq_len - 1:] = window_errors
    scores[:seq_len - 1] = window_errors[0]
    return scores


def detect_with_lstm(series, normal_end, seq_len, threshold_percentile, hidden_dim,
                     num_layers, epochs, lr, seed):
    """Train the autoencoder on `series[:normal_end]` only, then score the whole sequence.

    Returns `(scores, flags)` aligned to the timesteps of `series`.
    """
    series = np.asarray(series, dtype=float)
    train_windows = make_windows(series[:normal_end], seq_len)
    all_windows = make_windows(series, seq_len)

    model = train_lstm_autoencoder(train_windows, hidden_dim, num_layers, epochs, lr, seed)
    scores = _scores_to_timesteps(reconstruction_error(model, all_windows), len(series), seq_len)

    threshold = np.percentile(scores[:normal_end], threshold_percentile)
    return scores, scores > threshold
