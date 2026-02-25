## Dataset Description

- **Source**: Public electricity theft dataset (Kaggle) included as `data/raw/Electricity_Theft_Data.csv` + lightweight synthetic augmentation used for UI simulation.
- **Entities**: 9,957 customers (as recorded in `results/training_results.json`).
- **Time period**: 2014–2025 (as recorded in `results/training_results.json`).
- **Granularity**: Daily readings in the raw dataset.
- **Class imbalance**: Theft is a minority class.

### Labels / classes
- `0` = normal
- `1` = theft

### Synthetic / simulated components
This repository includes UI-level simulation to demonstrate a monitoring workflow. The dashboard’s “live” behavior should be treated as **near real-time (simulated)** monitoring unless connected to an actual streaming source.

### Included sample file
- `data/sample_data.csv` provides a small, human-readable sample (few hundred rows) to illustrate expected schema.

### Notes / limitations
- This is a prototype / research-style project.
- Utility-grade production deployment would require meter integration, stronger data validation, and a hardened security posture.
