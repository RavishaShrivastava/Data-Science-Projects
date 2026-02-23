
# Power Theft Detection in Smart Grids
---
## Overview

This repository contains a **working prototype** for detecting potential electricity theft in smart grids.

The system currently runs in **demo mode**, using **simulated smart-meter data** and **heuristic (rule-based) risk scoring** to demonstrate the full monitoring workflow — from data ingestion to alert generation and visualization.

The project is intentionally designed to be **ML-ready**, meaning real machine-learning models can be integrated later without changing the API or system structure.

---

## What this project demonstrates

* Monitoring electricity consumption patterns for anomalies
* Intrusion-detection–style workflow for smart meters
* Handling noisy or missing data gracefully
* Risk classification (**HIGH / MEDIUM / LOW / NORMAL**)
* A clear upgrade path to real ML inference

---

## Key Features

* **Simulated monitoring** via dashboard and API
* **Heuristic risk scoring (demo)** for anomaly detection
* **Rule-based alerts** with severity levels
* **Interactive web dashboard**
* **CSV dataset support** 
* **ML-ready architecture** (offline modules included)
* **CI + tests** using GitHub Actions

---

## Architecture (High Level)

```
Smart Meter Data (simulated / CSV)
        ↓
Anomaly & Risk Scoring (heuristic demo)
        ↓
Alerts & Logging
        ↓
Web Dashboard + API
```

---

## ML-Ready 

* Feature engineering and model prototypes exist under `src/`
* Training and experiments from the API runtime
* No real-time ML inference is wired in by default (by design)

This keeps the demo **honest and realistic**.

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

App runs at: `http://localhost:5000`

---

## Docker

```bash
docker build -t power-theft .
docker run -p 5000:5000 power-theft
```

---

## Limitations

* No real labeled theft dataset
* Risk scores are **heuristic**, not probabilities
* Dashboard metrics are illustrative

---


