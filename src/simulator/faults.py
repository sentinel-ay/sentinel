from __future__ import annotations

import numpy as np

FAULT_TYPES = ("bias", "drift", "dropout", "outlier")


def inject_fault(obs, fault_type, start, end, magnitude, rng, channels=(0, 1)):
    """Corrupt `obs[start:end, channels]` with one fault type.

    Returns `(corrupted, labels)` where labels is a 0/1 per-timestep ground truth mask.
    """
    obs = np.array(obs, dtype=float, copy=True)
    labels = np.zeros(len(obs), dtype=int)
    labels[start:end] = 1
    idx = np.ix_(np.arange(start, end), np.asarray(channels))
    span = end - start

    if fault_type == "bias":
        obs[idx] += magnitude
    elif fault_type == "drift":
        obs[idx] += np.linspace(0.0, magnitude, span)[:, None]
    elif fault_type == "dropout":
        obs[idx] = np.nan
    elif fault_type == "outlier":
        hits = rng.random(span) < 0.3
        signs = rng.choice([-1.0, 1.0], size=(span, len(channels)))
        obs[idx] += magnitude * signs * hits[:, None]
        labels[start:end] = hits.astype(int)      # only the spiked steps are anomalous
    else:
        raise ValueError(f"unknown fault_type: {fault_type!r} (expected one of {FAULT_TYPES})")

    return obs, labels
