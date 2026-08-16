import streamlit as st
import numpy as np

from src.estimation.baseline import simple_baseline
from src.estimation.kalman_filter import ConstantVelocityKalmanFilter
from src.simulator.trajectory import generate_trajectory

st.title("SENTINEL Demo")

steps = st.slider("Steps", 50, 500, 200)
traj = generate_trajectory(steps=steps)

baseline = simple_baseline(traj)

kf = ConstantVelocityKalmanFilter(dt=0.1, q=0.1, r=0.5)
filtered = kf.filter(traj)

st.line_chart(traj[:, :2])
st.line_chart(filtered[:, :2])

st.write({
    "ground_truth_points": traj.shape[0],
    "baseline_points": baseline.shape[0],
    "filtered_points": filtered.shape[0],
})