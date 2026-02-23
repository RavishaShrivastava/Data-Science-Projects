# Power Theft Detection in Smart Grids (Heuristic Demo, ML-Ready Architecture)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project is a **credible engineering demo** for power-theft monitoring in smart grids.

By default, the running dashboard/API operates in **simulated demo mode** and computes **heuristic risk scores** to demonstrate a **request-driven (poll-based) monitoring** workflow.

The repository also contains ML/feature-engineering modules intended as a foundation for a future “real inference” pipeline, but **real-time ML inference is not wired into the running API by default**.

## Scope
- Heuristic anomaly risk scoring + simulated monitoring in the dashboard
- ML-ready architecture (offline modules are present, but not connected to live inference)
- Research / prototype-level system

"ML-ready" in the sense that:

- feature extraction modules exist under `src/`
- training/experiments are decoupled from API runtime
- inference hooks can be added later without changing the API surface

## Non-Goals
- Hardware / meter firmware integration
- Utility-scale deployment guarantees
- Full production IAM / authorization

### 🎯 Objectives

- **Demonstrate an intrusion-detection workflow** for smart-meter consumption monitoring
- **Handle data-related challenges** effectively (missing values, class imbalance)
- **Provide a clear upgrade path** to real ML inference
- **Provide poll-based simulated monitoring** and automated alert generation
- **Support energy security** and grid sustainability

### ⚡ Key Features

- **Poll-based simulated monitoring**: Monitoring workflow demonstrated via dashboard + API
- **Heuristic Risk Scoring (Demo Mode)**: Produces risk-like outputs for UI/API demonstration
- **Dataset Support (Optional)**: Can load a CSV dataset if provided; otherwise simulates data
- **Interactive Dashboard**: Web-based monitoring interface
- **Rule-based alerts (demo)**: Threshold-based alert generation (HIGH/MEDIUM/LOW)
- **CI + basic tests**: Ruff + pytest run on every push via GitHub Actions

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Smart Meter Data (simulated / CSV-loaded readings - demo)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│     Anomaly-based Monitoring / Risk Scoring ("IDS")        │
│  • Heuristic risk scoring (demo mode)                       │
│  • Risk Classification (HIGH/MEDIUM/LOW)                    │
│  • Anomaly Detection                                        │
│  • Alert Generation & Management                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Web Dashboard & Monitoring                     │
│  • Poll-based simulated metrics                             │
│  • Alert Management                                         │
│  • Visualization & Reports                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚨 Intrusion Detection System

The term "IDS" is used here in a conceptual sense (monitoring abnormal consumption patterns), not as a traditional network IDS.

### Detection Process
1. **Data Collection**: Receive consumption reading
2. **Preprocessing**: Extract features
3. **Risk Scoring (Demo Mode)**: Compute a heuristic risk score
4. **Risk Classification**: Classify as HIGH/MEDIUM/LOW/NORMAL
5. **Alert Generation**: Create alerts for theft cases
6. **Logging**: Record all detections

## Upgrade Path to Real ML (Optional)

To convert this demo into a real ML inference system, you would typically add:
- A training script that saves a model artifact (e.g., `joblib` or Keras model)
- A saved scaler/normalizer and a stable feature list
- A deterministic preprocessing pipeline shared by training and inference
- API endpoints that load artifacts at startup and run real predictions on validated inputs

### Risk Levels
- **HIGH**: Risk score ≥ 0.80 (heuristic, non-probabilistic)
- **MEDIUM**: Risk score ≥ 0.50 (heuristic, non-probabilistic)
- **LOW**: Risk score ≥ 0.30 (heuristic, non-probabilistic)
- **NORMAL**: Risk score < 0.30 (heuristic, non-probabilistic)

## 📱 Web Dashboard Features

- **Poll-based simulated monitoring**: Live system status and metrics (simulated feed)
- **Detection Statistics**: Total detections and risk distribution (demo)
- **Alert Management**: View and acknowledge alerts
- **Risk Distribution**: Visualization of risk levels
- **Simulation Mode**: Test the system with random data
- **Export Functionality**: Download alerts and detection logs

## API (Demo)

All API routes are under `/api/*`.

If `API_KEY` is set, include:

- Header: `X-API-KEY: <your key>`

### `GET /api/available-years`
Response:

```json
{ "years": [2015, 2016, 2017] }
```

### `GET /api/year-detections/<year>`
Returns demo detection rows with a heuristic `risk_score`.

Invalid or missing readings are scored conservatively (demo) to simulate noisy field data rather than being rejected with `400`.

Response (array):

```json
[
  {
    "meter_id": "MTR-2024-00001",
    "risk_score": 0.83,
    "risk_level": "HIGH",
    "avg_consumption": 12.3,
    "expected_consumption": 22.0,
    "status": "Flagged",
    "detection_date": "2024-05-12",
    "year": 2024
  }
]
```

## API Key Protection

If `API_KEY` environment variable is set:

- All `/api/*` endpoints require header `X-API-KEY: <your key>`.
- Missing or wrong key returns `401`.

If `API_KEY` is not set:

- API routes are accessible after user login (demo mode).
- Login is a simple session-based demo login (no role management, no IAM).

## 📁 Project Structure

```
power-theft-detection-system/
├── src/
│   ├── data_preprocessing.py    # Data preprocessing and feature engineering
│   ├── models.py                # Experimental ML model prototypes (not used by demo runtime)
│   ├── intrusion_detection.py  # IDS implementation
│   ├── risk_scoring.py          # Heuristic risk score helper used by the demo runtime
│   └── visualization.py         # Plotting and visualization
├── templates/
│   ├── index_with_timeline.html # Web dashboard template
│   └── login.html              # Login page
├── data/
│   ├── raw/                     # (Optional) place larger datasets here
│   └── sample_data.csv          # Small illustrative sample
├── results/
│   ├── MODEL_PERFORMANCE_SUMMARY.md  # Offline/experimental notes
│   └── training_results.json         # Offline/experimental notes
├── tests/
│   ├── test_api.py
│   └── test_risk_scoring.py
├── app.py                      # Flask web application
├── .github/workflows/ci.yml     # CI (ruff + pytest)
├── Dockerfile
├── requirements.txt            # Runtime/demo dependencies
├── requirements-ml.txt         # Optional ML/visualization dependencies
└── README.md                   # This file
```

## Experimental / Offline Modules

The `src/` directory contains exploratory and prototype modules (feature engineering, model architectures, visualization helpers).
These are **not used by the current demo runtime** (`app.py`) by default.
They exist to document an ML-ready architecture and provide a starting point for future work.

## Dependencies

- `requirements.txt` is the lightweight set needed to run the demo web app.
- `requirements-ml.txt` contains optional heavy dependencies (TensorFlow, scikit-learn, plotting) for offline experiments.

## 🔧 Configuration

Configuration is done via environment variables (see Quick Start).

## 📊 Results & Visualization

The running demo focuses on the dashboard experience and simulated monitoring.

The `results/` artifacts and the visualization modules under `src/` are intended for offline experimentation.

## 🎓 Research References

These works informed feature ideas and system design; they are not directly implemented in the demo runtime.

## Deployment (Docker)

```bash
docker build -t power-theft .
docker run -p 5000:5000 power-theft
```

## Production Run (Gunicorn)

For a more production-like server, run with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Platform Procfile example:

```procfile
web: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

## Security Note

This is a prototype. If `API_KEY` is set in the environment, `/api/*` routes require:

- Header: `X-API-KEY: <your key>`

If `API_KEY` is not set, API routes remain accessible after login.

## Sample Dataset

This repo includes `data/sample_data.csv` as an illustrative example.

When using your own dataset, place it under `data/raw/` and set `DATASET_PATH` to point at it.

Typical columns you may want (example schema; your dataset can differ):

- `meter_id`
- `timestamp`
- `consumption`

## Limitations

- No labeled theft dataset is included in this demo.
- The heuristic risk score is not a calibrated probability.
- Dashboard metrics are illustrative and meant for UI/workflow demonstration.

## Testing

```bash
pytest -q
```

## CI

GitHub Actions runs on every push:

- `ruff check .`
- `pytest -q`

Based on literature survey including:

1. **Smart grids based on deep learning** (Noor Mahmoud Ibrahim et al., 2021)
2. **Electricity Theft Detection in Smart Grid Systems: A CNN-LSTM Based Approach** (Md. Nazmul Hasan et al., 2019)
3. **An Ensemble Deep Convolutional Neural Network Model for Electricity Theft Detection in Smart Grids** (Hossein Mohammadi Rouzbahani et al., 2021)
4. **An Intelligent Framework for Electricity Theft Detection in Smart Grid** (Yogesh Kulkarni et al., 2021)

