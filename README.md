# SENTINEL

가상 이동체와 다중 센서(GPS/IMU/Velocity/Distance) 환경을 시뮬레이션하고 센서 융합으로 상태를 추정(Kalman Filter)하며 센서 이상을 탐지(Statistical → ML → DL)하는 개인 R&D 프로젝트입니다.

실제 무기 설계/유도/추진/탄두는 다루지 않습니다.
일반적인 이동체와 센서 데이터를 대상으로 합니다.

## 핵심 설계

- KF state vector: `[px, py, vx, vy]` (constant velocity 모델). Ground truth 궤적은 IMU(가속도계) 관측을 위해 `[px, py, vx, vy, ax, ay]`로 기록
- 센서 파라미터(noise_std, bias, drift_rate, dropout_prob, outlier_prob)는 전부 [config/config.yaml](config/config.yaml)에서 관리, 하드코딩 없음
- 핵심 알고리즘(Kalman Filter)은 라이브러리 대체 없이 직접 구현. 프로세스 잡음 `Q`는 매직넘버가 아니라 이산 백색잡음 가속도 모델에서 `dt`로 유도
- 센서 결측(dropout)은 KF `update()` 단계에서 해당 관측 행을 마스킹해 predict만 수행
- 이상탐지 3종 비교: Statistical(z-score) / Isolation Forest / LSTM Autoencoder — 전부 **KF 잔차(innovation)** 기반
- 평가지표: State Estimation은 MAE/RMSE, Anomaly Detection은 Precision/Recall/F1/ROC-AUC

## 구조

```
sentinel/
├── config/config.yaml       # 시뮬레이터/센서/추정/이상탐지 파라미터 (단일 소스)
├── requirements.txt
├── src/
│   ├── config.py            # config.yaml 로더
│   ├── simulator/           # trajectory.py (궤적 5종), faults.py (이상 주입)
│   ├── sensors/             # base_sensor, gps, imu, velocity, distance, factory
│   ├── preprocessing/       # forward-fill 결측 처리, 센서 DataFrame 구성
│   ├── estimation/          # baseline.py, kalman_filter.py (직접 구현)
│   ├── anomaly_detection/   # statistical / ml_detector / dl_detector
│   ├── models/              # lstm_autoencoder.py (sliding window)
│   └── evaluation/          # MAE/RMSE, Precision/Recall/F1, ROC-AUC
├── dashboard/app.py         # Streamlit 대시보드
├── experiments/             # experiment_0N_*.py
├── tests/
└── docs/
```

## 진행 상황

- [x] Phase 1 — 시뮬레이터 + 센서 데이터 생성 (궤적 5종: straight/curve/accelerate/decelerate/turn)
- [x] Phase 2 — Baseline + Kalman Filter
- [x] Phase 3 — Anomaly Detection (Statistical → ML → DL)
- [x] Phase 4 — Deep Learning (LSTM Autoencoder, sliding window)
- [x] Phase 5 — 성능 평가 (Baseline vs KF, 이상탐지 3종 비교)
- [x] Phase 6 — Streamlit 대시보드
- [ ] Phase 7 — 문서화

## 결과

### 상태 추정 (steps=200, straight, seed=42)

| method | MAE | RMSE |
|---|---|---|
| raw obs (forward-fill) | 0.1926 | 0.2517 |
| baseline (직전 관측 유지) | 0.1903 | 0.2520 |
| **Kalman Filter** | **0.0620** | **0.0884** |

KF가 원시 관측 대비 MAE 68% 감소. 200 스텝 중 7 스텝에서 dropout 발생.

### 이상 탐지 (steps=600, 이상 주입 구간 [400,500), 정상 구간 [0,400)으로만 학습·캘리브레이션)

| 시나리오 | Statistical (F1/AUC) | Isolation Forest (F1/AUC) | LSTM AE (F1/AUC) |
|---|---|---|---|
| bias | 0.319 / 0.719 | 0.352 / 0.833 | **0.550 / 0.842** |
| drift | 0.364 / 0.836 | 0.395 / 0.829 | **0.476** / 0.772 |
| dropout | **0.696** / 0.845 | 0.556 / **0.867** | 0.541 / 0.798 |
| outlier | **0.892 / 1.000** | 0.828 / 0.995 | 0.328 / 0.923 |

점(point) 이상인 outlier는 단순 z-score가 가장 강하고, 지속형 이상(bias/drift)은 시퀀스 문맥을 보는 LSTM AE가 우세합니다.

**해석 시 유의점**

- LSTM AE는 `seq_len=50` 윈도우 단위로 점수를 내므로 이상 종료 후 최대 49스텝까지 탐지가 번집니다. outlier에서 Recall 1.000·Precision 0.196인 이유가 이것으로, 지속형 이상에 강한 대신 점 이상의 위치 특정에는 약합니다.
- GPS가 유일한 위치 관측원이라 지속적 위치 bias는 KF가 결국 따라가면서 잔차가 0으로 수렴합니다. bias 탐지 성능 상한이 여기서 결정되며, 독립적인 위치 관측(distance 센서의 EKF 융합)을 추가해야 근본적으로 개선됩니다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

pytest -q                                          # 테스트 23개
python -m experiments.experiment_01_simulation     # 센서 데이터 생성 + dropout 확인
python -m experiments.experiment_02_estimation     # KF vs baseline (MAE/RMSE + 시각 비교)
python -m experiments.experiment_03_anomaly        # 이상탐지 3종 비교표
python -m experiments.experiment_04_trajectory_types   # 궤적 5종 시각화 -> data/generated/
streamlit run dashboard/app.py                     # 대시보드
```

## 기술 스택

Python, NumPy, Pandas, SciPy, scikit-learn, PyTorch, Matplotlib, Streamlit