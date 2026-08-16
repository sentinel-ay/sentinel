# SENTINEL

가상 이동체와 다중 센서(GPS/IMU/Velocity/Distance) 환경을 시뮬레이션하고 센서 융합으로 상태를 추정(Kalman Filter)하며 센서 이상을 탐지(Statistical → ML → DL)하는 개인 R&D 프로젝트입니다.

실제 무기 설계/유도/추진/탄두는 다루지 않습니다.
일반적인 이동체와 센서 데이터를 대상으로 합니다.

## 핵심 설계

- State vector: `[px, py, vx, vy]` (constant velocity 모델)
- 센서 파라미터(noise_std, bias, drift_rate, dropout_prob, outlier_prob)는 전부 [config/config.yaml](config/config.yaml)에서 관리, 하드코딩 없음
- 핵심 알고리즘(Kalman Filter)은 라이브러리 대체 없이 직접 구현
- 이상탐지 3종 비교 예정: Statistical(z-score) / Isolation Forest / LSTM Autoencoder
- 평가지표: State Estimation은 MAE/RMSE, Anomaly Detection은 Precision/Recall/F1/ROC-AUC

## 구조

```
sentinel/
├── config/config.yaml       # 시뮬레이터/센서/추정/이상탐지 파라미터
├── src/
│   ├── simulator/           # 궤적 생성
│   ├── sensors/             # GPS, IMU, Velocity, Distance 센서 모델
│   ├── preprocessing/       # 전처리 파이프라인
│   ├── estimation/          # Baseline, Kalman Filter
│   ├── anomaly_detection/   # Statistical / ML / DL 이상탐지
│   ├── models/              # LSTM Autoencoder
│   └── evaluation/          # 평가지표
├── dashboard/app.py         # Streamlit 대시보드
├── experiments/             # experiment_0N_*.py
├── tests/
└── docs/
```

## 진행 상황

- [x] Phase 1 — 시뮬레이터 + 센서 데이터 생성
- [x] Phase 2 — Baseline + Kalman Filter
- [ ] Phase 3 — Anomaly Detection (Statistical → ML → DL)
- [ ] Phase 4 — Deep Learning (LSTM Autoencoder)
- [ ] Phase 5 — 성능 평가 (Baseline vs KF vs ML vs DL)
- [ ] Phase 6 — Streamlit 대시보드
- [ ] Phase 7 — 문서화

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib pyyaml pytest

pytest -q                                    # 테스트
python -m experiments.experiment_01_simulation
python -m experiments.experiment_02_estimation   # KF vs baseline 시각 비교
```

## 기술 스택

Python, NumPy, Pandas, SciPy, scikit-learn, PyTorch, Matplotlib, Streamlit