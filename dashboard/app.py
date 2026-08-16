import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `streamlit run` puts dashboard/ on the path, not the repo root

from src.anomaly_detection.dl_detector import detect_with_lstm
from src.anomaly_detection.ml_detector import anomaly_scores, fit_isolation_forest
from src.anomaly_detection.statistical import zscore_anomaly_scores
from src.config import load_config
from src.estimation.baseline import simple_baseline
from src.estimation.kalman_filter import build_kalman_filter
from src.evaluation.metrics import mae_rmse, precision_recall_f1, roc_auc
from src.preprocessing.pipeline import forward_fill
from src.sensors.factory import build_sensors
from src.simulator.faults import FAULT_TYPES, inject_fault
from src.simulator.trajectory import TRAJECTORY_TYPES, generate_from_config

st.set_page_config(page_title="SENTINEL", layout="wide")
cfg = load_config()
seed = cfg["project"]["random_seed"]

st.sidebar.header("Simulation")
traj_type = st.sidebar.selectbox("Trajectory", TRAJECTORY_TYPES)
steps = st.sidebar.slider("Steps", 200, 1000, cfg["anomaly"]["steps"], step=50)

st.sidebar.header("Injected fault")
fault = st.sidebar.selectbox("Type", ("none",) + FAULT_TYPES)
fault_start = st.sidebar.slider("Start", 0, steps - 1, min(cfg["anomaly"]["fault_start"], steps - 1))
fault_len = st.sidebar.slider("Length", 10, 300, 100)
detector = st.sidebar.selectbox("Detector", ("Statistical (z-score)", "Isolation Forest", "LSTM Autoencoder"))


@st.cache_data(show_spinner=False)
def simulate(traj_type, steps, seed):
    sim = dict(cfg["simulator"], steps=steps, traj_type=traj_type)
    rng = np.random.default_rng(seed)
    traj = generate_from_config({**cfg, "simulator": sim}, rng=rng)
    sensors = build_sensors(cfg, rng=rng)
    dt = sim["dt"]
    readings = {n: np.array([s.measure(state, i * dt) for i, state in enumerate(traj)])
                for n, s in sensors.items()}
    obs = np.column_stack([readings["gps"], readings["velocity"]])
    return traj, obs, readings


@st.cache_data(show_spinner="Running detector…")
def detect(detector, residuals, normal_end, seed):
    a = cfg["anomaly"]
    normal = residuals[:normal_end]
    if detector == "Statistical (z-score)":
        return zscore_anomaly_scores(residuals, a["zscore_threshold"], reference=normal)
    if detector == "Isolation Forest":
        forest = fit_isolation_forest(normal, a["contamination"], seed)
        scores = anomaly_scores(forest, residuals)
        return scores, scores > np.percentile(anomaly_scores(forest, normal), a["threshold_percentile"])
    return detect_with_lstm(residuals, normal_end=normal_end, seq_len=a["seq_len"],
                            threshold_percentile=a["threshold_percentile"], seed=seed, **a["lstm"])


traj, obs_clean, readings = simulate(traj_type, steps, seed)
truth = traj[:, :4]

fault_end = min(fault_start + fault_len, steps)
if fault == "none":
    obs, labels = obs_clean, np.zeros(steps, dtype=int)
else:
    obs, labels = inject_fault(obs_clean, fault, fault_start, fault_end,
                               cfg["anomaly"]["magnitude"][fault], np.random.default_rng(seed))

filled = forward_fill(obs)
kf_est, residuals = build_kalman_filter(cfg).filter(filled, return_residuals=True)
baseline = simple_baseline(filled)

normal_end = fault_start if fault != "none" and fault_start >= cfg["anomaly"]["seq_len"] else steps
scores, flags = detect(detector, residuals, normal_end, seed)

st.title("SENTINEL — sensor fusion & anomaly detection")

c1, c2, c3, c4 = st.columns(4)
c1.metric("KF MAE", f"{mae_rmse(kf_est, truth)[0]:.4f}")
c2.metric("Baseline MAE", f"{mae_rmse(baseline, truth)[0]:.4f}")
c3.metric("Raw obs MAE", f"{mae_rmse(filled, truth)[0]:.4f}")
c4.metric("Flagged steps", f"{int(flags.sum())} / {steps}")

left, right = st.columns(2)
with left:
    st.subheader("Trajectory")
    st.caption("Ground truth vs noisy GPS vs Kalman estimate")
    st.scatter_chart(
        pd.DataFrame({"px": np.concatenate([truth[:, 0], filled[:, 0], kf_est[:, 0]]),
                      "py": np.concatenate([truth[:, 1], filled[:, 1], kf_est[:, 1]]),
                      "series": ["ground truth"] * steps + ["gps obs"] * steps + ["kalman"] * steps}),
        x="px", y="py", color="series", height=380)

with right:
    st.subheader(f"Anomaly score — {detector}")
    st.caption("Score per timestep, with flagged steps and the injected fault window")
    chart = pd.DataFrame({"score": scores, "flagged": np.where(flags, scores, np.nan)})
    if fault != "none":
        chart["injected fault"] = np.where(labels == 1, np.nanmax(scores), np.nan)
    st.line_chart(chart, height=380)

if fault != "none":
    p, r, f1 = precision_recall_f1(labels, flags)
    st.subheader("Detection performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{p:.3f}")
    m2.metric("Recall", f"{r:.3f}")
    m3.metric("F1", f"{f1:.3f}")
    m4.metric("ROC-AUC", f"{roc_auc(labels, scores):.3f}")

st.subheader("Sensor status")
status = pd.DataFrame([
    {"sensor": name,
     "channels": np.atleast_2d(v.T).shape[0],
     "dropouts": int(np.isnan(v).sum()),
     "dropout rate": f"{np.isnan(v).mean():.2%}",
     "configured dropout_prob": cfg["sensors"][name]["dropout_prob"],
     "status": "DEGRADED" if np.isnan(v).mean() > 2 * cfg["sensors"][name]["dropout_prob"] else "OK"}
    for name, v in readings.items()])
st.dataframe(status, width="stretch", hide_index=True)
