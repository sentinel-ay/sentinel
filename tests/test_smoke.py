import numpy as np
import pytest

from src.config import load_config
from src.estimation.baseline import simple_baseline
from src.estimation.kalman_filter import build_kalman_filter
from src.preprocessing.pipeline import forward_fill
from src.sensors.factory import build_sensors
from src.anomaly_detection.statistical import zscore_anomaly_scores
from src.models.lstm_autoencoder import make_windows
from src.simulator.faults import FAULT_TYPES, inject_fault
from src.simulator.trajectory import TRAJECTORY_TYPES, generate_from_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_has_every_section(cfg):
    assert {"simulator", "sensors", "estimation", "anomaly", "project"} <= cfg.keys()
    assert {"gps", "imu", "velocity", "distance"} == cfg["sensors"].keys()


def test_trajectory_shape_and_finiteness(cfg):
    traj = generate_from_config(cfg, rng=np.random.default_rng(42))
    assert traj.shape == (cfg["simulator"]["steps"], 6)
    assert np.all(np.isfinite(traj))

    baseline = simple_baseline(traj)
    assert baseline.shape == traj.shape
    assert np.all(np.isfinite(baseline))


@pytest.mark.parametrize("traj_type", TRAJECTORY_TYPES)
def test_every_trajectory_type_runs(cfg, traj_type):
    traj = generate_from_config(cfg, rng=np.random.default_rng(42), traj_type=traj_type)
    assert traj.shape == (cfg["simulator"]["steps"], 6)
    assert np.all(np.isfinite(traj))


def test_unknown_trajectory_type_rejected(cfg):
    with pytest.raises(ValueError):
        generate_from_config(cfg, rng=np.random.default_rng(42), traj_type="spiral")


def test_drift_grows_proportionally_with_time(cfg):
    """Task 3: drift must accumulate with elapsed time, not with the measured magnitude."""
    sensor = build_sensors(cfg, rng=np.random.default_rng(0))["gps"]
    sensor.dropout_prob = 0.0
    sensor.outlier_prob = 0.0

    def mean_error(t, n=20000):
        return np.mean([sensor.sample(10.0, t) - 10.0 for _ in range(n)])

    e0, e1000 = mean_error(0.0), mean_error(1000.0)
    expected = sensor.drift_rate * 1000.0

    assert e0 == pytest.approx(sensor.bias, abs=0.02)
    assert (e1000 - e0) == pytest.approx(expected, rel=0.05)


def test_drift_independent_of_measured_value(cfg):
    """The old formula scaled drift by `value`; the same value must now not change the error."""
    sensor = build_sensors(cfg, rng=np.random.default_rng(0))["gps"]
    sensor.dropout_prob = 0.0
    sensor.outlier_prob = 0.0
    sensor.noise_std = 0.0

    assert sensor.sample(1.0, 5.0) - 1.0 == pytest.approx(sensor.sample(1000.0, 5.0) - 1000.0)


def test_imu_and_velocity_measure_different_quantities(cfg):
    """Task 4: IMU is an accelerometer, it must not mirror the velocity sensor."""
    sensors = build_sensors(cfg, rng=np.random.default_rng(1))
    for s in sensors.values():
        s.noise_std = s.bias = s.drift_rate = s.dropout_prob = s.outlier_prob = 0.0

    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(sensors["velocity"].measure(state, 0.0), [3.0, 4.0])
    assert np.allclose(sensors["imu"].measure(state, 0.0), [5.0, 6.0])
    assert np.allclose(sensors["gps"].measure(state, 0.0), [1.0, 2.0])
    assert sensors["distance"].measure(state, 0.0) == pytest.approx(np.hypot(1.0, 2.0))


def test_forward_fill_removes_dropouts():
    values = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan], [4.0, 5.0]])
    filled = forward_fill(values)
    assert not np.isnan(filled).any()
    assert np.allclose(filled[:, 0], [2.0, 2.0, 2.0, 4.0])   # leading gap back-filled
    assert np.allclose(filled[:, 1], [1.0, 1.0, 1.0, 5.0])


def test_kalman_skips_update_on_missing_observation(cfg):
    kf = build_kalman_filter(cfg)
    obs = np.array([[0.0, 0.0, 1.0, 0.5],
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.2, 0.1, 1.0, 0.5]])
    est = kf.filter(obs)
    assert est.shape == (3, 4)
    assert np.all(np.isfinite(est))     # a fully-dropped step must not poison the state


def test_kalman_beats_raw_observations(cfg):
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    traj = generate_from_config(cfg, rng=rng)
    sensors = build_sensors(cfg, rng=rng)
    dt = cfg["simulator"]["dt"]
    obs = np.array([np.concatenate([sensors["gps"].measure(s, i * dt),
                                    sensors["velocity"].measure(s, i * dt)])
                    for i, s in enumerate(traj)])

    from src.evaluation.metrics import mae_rmse
    truth = traj[:, :4]
    kf_mae, _ = mae_rmse(build_kalman_filter(cfg).filter(obs), truth)
    raw_mae, _ = mae_rmse(forward_fill(obs), truth)
    assert kf_mae < raw_mae


def test_kalman_Q_is_derived_from_dt(cfg):
    """Q must come from the white-noise acceleration model, not a scalar on the diagonal."""
    kf = build_kalman_filter(cfg)
    dt, q = cfg["simulator"]["dt"], cfg["estimation"]["accel_process_std"]
    assert kf.Q[0, 0] == pytest.approx(q ** 2 * dt ** 4 / 4)
    assert kf.Q[2, 2] == pytest.approx(q ** 2 * dt ** 2)
    assert kf.Q[0, 2] == pytest.approx(q ** 2 * dt ** 3 / 2)   # position/velocity correlated


@pytest.mark.parametrize("fault", FAULT_TYPES)
def test_inject_fault_labels_and_shape(fault, cfg):
    obs = np.zeros((300, 4))
    corrupted, labels = inject_fault(obs, fault, 100, 200, 3.0, np.random.default_rng(0))
    assert corrupted.shape == obs.shape
    assert labels.sum() > 0
    assert labels[:100].sum() == 0 and labels[200:].sum() == 0     # fault confined to the window
    assert np.allclose(corrupted[:100], 0.0)                       # prefix untouched
    assert np.allclose(corrupted[:, 2:], 0.0)                      # velocity channels untouched
    if fault == "dropout":
        assert np.isnan(corrupted[100:200, :2]).all()


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        inject_fault(np.zeros((10, 4)), "explode", 1, 5, 1.0, np.random.default_rng(0))


def test_make_windows_shape():
    w = make_windows(np.arange(20).reshape(10, 2), seq_len=4)
    assert w.shape == (7, 4, 2)
    assert np.allclose(w[0], np.arange(8).reshape(4, 2))
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 2)), seq_len=10)


def test_zscore_calibrates_on_reference():
    values = np.concatenate([np.zeros(100), np.full(10, 50.0)])
    _, flags = zscore_anomaly_scores(values, 3.0, reference=np.random.default_rng(0).normal(0, 1, 500))
    assert flags[100:].all() and not flags[:100].any()


def test_detectors_return_timestep_aligned_output(cfg):
    """Every detector must emit one score/flag per timestep of the input sequence."""
    from experiments.experiment_03_anomaly import run_detectors
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1, (200, 4))
    for name, (scores, flags) in run_detectors(cfg, residuals, normal_end=150, seed=0).items():
        assert scores.shape == (200,), name
        assert flags.shape == (200,), name
        assert flags.dtype == bool, name
