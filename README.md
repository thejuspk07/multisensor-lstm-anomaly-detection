# Multi-Sensor LSTM Anomaly Detection

A machine-learning project for detecting abnormal behaviour in turbofan engine sensor data using **LSTM autoencoders**, reconstruction-error analysis, and a Flask dashboard.

## Overview

The application works with the NASA CMAPSS-style multi-engine dataset. It processes engine sensor measurements, builds time-series sequences, estimates reconstruction error, and presents engine health and anomaly information through a web dashboard.

The current application also includes a demo mode that loads the dataset and produces smoothed anomaly/error curves without requiring LSTM training at startup.

## Features

- Multi-engine sensor-data processing
- LSTM autoencoder architecture
- Reconstruction-error based anomaly detection
- Engine health comparison
- Anomaly cycle detection
- Remaining Useful Life (RUL) estimation
- Flask web dashboard
- Matplotlib visualizations
- Demo mode for quick exploration
- Modular data-loading and model utilities

## Project Structure

```text
multisensor-lstm-anomaly-detection/
├── anomaly_detection.py     # Reconstruction error, anomaly and RUL utilities
├── app.py                   # Flask dashboard application
├── data_loader.py           # Dataset loading and preprocessing
├── model.py                 # LSTM autoencoder definition
├── datasets/                # CMAPSS datasets
├── templates/               # Dashboard HTML templates
├── static/                  # Static dashboard assets
├── requirements.txt         # Python dependencies
└── README.md
```

## Workflow

```text
CMAPSS Sensor Data
        │
        ▼
Data Loading & Selection
        │
        ▼
Normalization
        │
        ▼
Time-Series Sequences
        │
        ▼
LSTM Autoencoder
        │
        ▼
Reconstruction Error
        │
   ┌────┴────┐
   ▼         ▼
Normal    Anomaly
   │         │
   └────┬────┘
        ▼
Health / RUL Dashboard
```

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Flask
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/thejuspk07/multisensor-lstm-anomaly-detection.git
cd multisensor-lstm-anomaly-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The dashboard processes the configured dataset and displays engine-health comparisons, reconstruction-error trends, anomalous cycles, and RUL-related information.

## Model

The project uses an LSTM autoencoder for time-series anomaly detection. Sensor sequences are normalized before being passed to the model. Reconstruction error is used as the anomaly signal: higher error indicates that the observed behaviour differs from patterns learned by the model.

The code also provides utilities for estimating Remaining Useful Life from the observed error trend.

## Dataset

The project is designed around NASA CMAPSS turbofan engine data and expects the relevant files under `datasets/`. The repository includes dataset-loading utilities for working with multiple engines.

## Demo Mode

`app.py` currently enables a demo mode by default. In this mode, the dashboard loads engine data and generates a normalized, smoothed error signal without training an LSTM model at startup. This makes the application easier to run for demonstrations.

## Research Direction

Potential improvements include:

- Training one global model on multiple engines
- Better threshold calibration
- More robust RUL estimation
- Cross-engine validation
- Sensor-selection experiments
- Real-time streaming support
- Model explainability
- Automated model evaluation

## Author

**Thejus P. K.**  
GitHub: https://github.com/thejuspk07
