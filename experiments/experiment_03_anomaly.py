"""Statistical vs Isolation Forest vs LSTM Autoencoder on KF residuals.

Protocol (identical for all three, so the comparison is fair): every detector is
fitted/calibrated on the known-normal prefix `[:fault_start]` only, then scores the
full sequence. Features are the Kalman innovation `z - H x_pred`, not raw measurements.
"""
import numpy as np

from src.anomaly_detection.dl_detector import detect_with_lstm
from src.anomaly_detection.ml_detector import anomaly_scores, fit_isolation_forest
from src.anomaly_detection.statistical import zscore_anomaly_scores
from src.config import load_config
from src.estimation.kalman_filter import build_kalman_filter
from src.evaluation.metrics import precision_recall_f1, roc_auc
from src.preprocessing.pipeline import forward_fill
from src.sensors.factory import build_sensors
from src.simulator.faults import FAULT_TYPES, inject_fault
from src.simulator.trajectory import generate_from_config


def clean_observations(cfg, rng):
    sim = dict(cfg["simulator"], steps=cfg["anomaly"]["steps"])
    traj = generate_from_config({**cfg, "simulator": sim}, rng=rng)
    sensors = build_sensors(cfg, rng=rng)
    dt = sim["dt"]
    return np.array([np.concatenate([sensors["gps"].measure(s, i * dt),
                                     sensors["velocity"].measure(s, i * dt)])
                     for i, s in enumerate(traj)])


def kf_residuals(cfg, obs):
    """Kalman innovation on forward-filled observations (a dropout shows up as a frozen reading)."""
    _, residuals = build_kalman_filter(cfg).filter(forward_fill(obs), return_residuals=True)
    return residuals


def run_detectors(cfg, residuals, normal_end, seed):
    a = cfg["anomaly"]
    normal = residuals[:normal_end]

    z_scores, z_flags = zscore_anomaly_scores(residuals, a["zscore_threshold"], reference=normal)

    forest = fit_isolation_forest(normal, a["contamination"], seed)
    if_scores = anomaly_scores(forest, residuals)
    if_flags = if_scores > np.percentile(anomaly_scores(forest, normal), a["threshold_percentile"])

    lstm_scores, lstm_flags = detect_with_lstm(
        residuals, normal_end=normal_end, seq_len=a["seq_len"],
        threshold_percentile=a["threshold_percentile"], seed=seed, **a["lstm"])

    return {"Statistical (z-score)": (z_scores, z_flags),
            "Isolation Forest": (if_scores, if_flags),
            "LSTM Autoencoder": (lstm_scores, lstm_flags)}


if __name__ == "__main__":
    cfg = load_config()
    a = cfg["anomaly"]
    seed = cfg["project"]["random_seed"]
    obs_clean = clean_observations(cfg, np.random.default_rng(seed))

    print(f"steps={a['steps']}  fault window=[{a['fault_start']}, {a['fault_end']})  "
          f"seq_len={a['seq_len']}  train on [0, {a['fault_start']})\n")
    header = f"{'scenario':<10}{'detector':<24}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>10}"

    for fault in FAULT_TYPES:
        obs, labels = inject_fault(obs_clean, fault, a["fault_start"], a["fault_end"],
                                   a["magnitude"][fault], np.random.default_rng(seed))
        residuals = kf_residuals(cfg, obs)
        results = run_detectors(cfg, residuals, a["fault_start"], seed)

        print(header)
        print("-" * len(header))
        for name, (scores, flags) in results.items():
            p, r, f1 = precision_recall_f1(labels, flags)
            print(f"{fault:<10}{name:<24}{p:>11.3f}{r:>9.3f}{f1:>8.3f}{roc_auc(labels, scores):>10.3f}")
        print()
