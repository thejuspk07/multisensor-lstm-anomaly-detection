# Multi-Sensor LSTM Anomaly Detection

An end-to-end time-series anomaly detection project for turbofan engine sensor data using **LSTM autoencoders**, reconstruction error, and a Flask-based visualization dashboard.

## 🎯 Project Goal

Modern machines produce large amounts of sensor data. The challenge is identifying when sensor behaviour starts to differ from normal operating patterns.

This project explores that problem using NASA CMAPSS-style turbofan engine data. Sensor measurements are transformed into time-series sequences, processed by an LSTM autoencoder, and analyzed using reconstruction error to identify unusual engine behaviour.

> **Core idea:** If an LSTM autoencoder learns normal sensor behaviour well, an unusual sequence should produce a larger reconstruction error.

## ✨ Features

- Multi-engine turbofan sensor-data processing
- Time-series sequence preparation
- LSTM autoencoder architecture
- Reconstruction-error based anomaly detection
- Anomaly-cycle analysis
- Engine health comparison
- Remaining Useful Life (RUL) estimation utilities
- Flask web dashboard
- Matplotlib visualizations
- Demo mode for quick exploration
- Modular data-loading and model utilities

## 🧠 How It Works

```text
NASA CMAPSS Sensor Data
          │
          ▼
   Data Loading
          │
          ▼
 Sensor Selection / Cleaning
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
  Normal    Anomalous
  Pattern    Pattern
     │         │
     └────┬────┘
          ▼
 Engine Health / RUL Analysis
          │
          ▼
     Flask Dashboard
```

## 📁 Project Structure

```text
multisensor-lstm-anomaly-detection/
├── anomaly_detection.py     # Anomaly, reconstruction-error and RUL utilities
├── app.py                   # Flask dashboard
├── data_loader.py           # Dataset loading and preprocessing
├── model.py                 # LSTM autoencoder definition
├── datasets/                # CMAPSS-style datasets
├── templates/               # Flask HTML templates
├── static/                  # Dashboard assets
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| TensorFlow / Keras | LSTM autoencoder |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Scikit-learn | Preprocessing and ML utilities |
| Matplotlib | Error and health visualizations |
| Flask | Web dashboard |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/thejuspk07/multisensor-lstm-anomaly-detection.git
cd multisensor-lstm-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
python app.py
```

Open the application at:

```text
http://127.0.0.1:5000
```

## 📊 Anomaly Detection Concept

The LSTM autoencoder learns to reconstruct time-series sensor sequences. The difference between the original sequence and its reconstruction is represented by the **reconstruction error**.

Conceptually:

```text
Low reconstruction error  → behaviour similar to learned patterns
High reconstruction error → potentially abnormal behaviour
```

A threshold can then be used to separate normal and potentially anomalous observations. In a production system, this threshold should be calibrated using validation data rather than chosen arbitrarily.

## 🧪 Demo Mode

The Flask application currently includes a demo mode designed for easier project demonstrations. It can load engine data and generate a normalized, smoothed error signal without requiring LSTM training every time the dashboard starts.

This makes the project convenient for experimentation and presentation while keeping the LSTM model as the core research direction.

## 📦 Dataset

The project is designed around **NASA CMAPSS-style turbofan engine datasets**. Relevant dataset files are expected under the `datasets/` directory.

The data represents run-to-failure behaviour from simulated turbofan engines and is commonly used for predictive-maintenance and Remaining Useful Life research.

## 🔬 Research / Future Improvements

Possible next steps include:

- Train a global model across multiple engines
- Establish data-driven anomaly thresholds
- Add cross-engine validation
- Improve RUL estimation
- Compare different sensor subsets
- Add real-time sensor streaming
- Add model evaluation metrics
- Add explainability for detected anomalies
- Add experiment tracking
- Containerize the Flask application

## ⚠️ Limitations

- Demo mode does not represent a fully trained production anomaly detector.
- Anomaly thresholds require proper validation and calibration.
- RUL estimation is experimental and should not be treated as an operational prediction without further validation.
- CMAPSS is a simulated dataset, so real-world deployment would require domain-specific validation.

## 👤 Author

**Thejus P. K.**

GitHub: https://github.com/thejuspk07

---

⭐ If you find this project useful, consider giving the repository a star.
