# Multisensor LSTM Anomaly Detection

A web-based application for detecting anomalies in multisensor time-series data using Long Short-Term Memory (LSTM) neural networks. This project implements an unsupervised anomaly detection approach suitable for industrial IoT, equipment monitoring, and sensor data analysis.

## Overview

This application uses LSTM-based deep learning to identify abnormal patterns in multivariate sensor data. The system learns normal operational behavior from training data and detects deviations that may indicate equipment failures, malfunctions, or unusual events.

## Features

- **LSTM-Based Anomaly Detection**: Leverages recurrent neural networks to capture temporal dependencies in sensor data
- **Multi-Sensor Support**: Handles multiple sensor inputs simultaneously for comprehensive system monitoring
- **Web Interface**: User-friendly Flask-based web application for data upload, model training, and visualization
- **Real-Time Visualization**: Interactive charts and plots for detected anomalies
- **Customizable Threshold**: Adjustable sensitivity for anomaly detection based on use case requirements

## Project Structure

```
multisensor-lstm-anomaly-detection/
├── datasets/              # Sample datasets and data storage
├── static/                # CSS, JavaScript, and static assets
├── templates/             # HTML templates for web interface
├── anomaly_detection.py   # Core anomaly detection algorithms
├── app.py                 # Flask web application
├── data_loader.py         # Data preprocessing and loading utilities
├── model.py               # LSTM model architecture and training
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore configuration
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thejuspk07/multisensor-lstm-anomaly-detection.git
cd multisensor-lstm-anomaly-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the Flask web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload your sensor data (CSV format recommended)

4. Configure model parameters and train the LSTM network

5. View anomaly detection results and visualizations

### Data Format

Input data should be in CSV format with the following structure:
- First column: Timestamp (optional)
- Subsequent columns: Sensor readings (numerical values)
- No missing values (preprocessing required beforehand)

Example:
```csv
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,1.23,4.56,7.89
2024-01-01 00:01:00,1.25,4.58,7.91
...
```

## Model Architecture

The system employs an LSTM-based architecture for time-series anomaly detection:

1. **Input Layer**: Accepts sequential multisensor data
2. **LSTM Layers**: Captures temporal patterns and dependencies
3. **Dense Layers**: Learns representations for reconstruction/prediction
4. **Output Layer**: Generates predictions or reconstructions
5. **Anomaly Scoring**: Calculates reconstruction error or prediction deviation

### Training Process

1. **Data Preprocessing**: Normalization and sequence creation
2. **Model Training**: LSTM learns normal behavior patterns
3. **Threshold Calculation**: Statistical analysis of training errors
4. **Anomaly Detection**: Flags points exceeding threshold as anomalies

## Configuration

Key parameters can be adjusted in the respective modules:

- **Sequence Length**: Number of time steps for LSTM input
- **LSTM Units**: Hidden layer dimensions
- **Learning Rate**: Training optimization parameter
- **Batch Size**: Number of samples per training iteration
- **Epochs**: Number of complete passes through training data
- **Threshold Multiplier**: Sensitivity of anomaly detection

## Technology Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow/Keras or PyTorch (check requirements.txt)
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly (check static/templates)
- **Frontend**: HTML, CSS, JavaScript

## Use Cases

- Industrial equipment monitoring
- IoT sensor data analysis
- Predictive maintenance
- Quality control in manufacturing
- Environmental monitoring systems
- Network traffic analysis
- Healthcare vital sign monitoring

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is available for educational and research purposes. Please check with the repository owner for specific licensing terms.

## Acknowledgments

- Inspired by research in LSTM-based anomaly detection for multivariate time series
- Based on encoder-decoder architectures for unsupervised learning
- References seminal work in time-series anomaly detection using deep learning

## Contact

**Jus PK**
- GitHub: [@thejuspk07](https://github.com/thejuspk07)

## Future Enhancements

- [ ] Support for real-time streaming data
- [ ] Multiple model architectures (GRU, Transformer, Autoencoder variants)
- [ ] Automated hyperparameter tuning
- [ ] Model performance metrics dashboard
- [ ] Export functionality for detected anomalies
- [ ] REST API for integration with other systems
- [ ] Docker containerization
- [ ] Cloud deployment guides

## Troubleshooting

**Issue**: Model not converging
- **Solution**: Adjust learning rate, increase epochs, or normalize data

**Issue**: Too many false positives
- **Solution**: Increase threshold multiplier or retrain with more diverse data

**Issue**: Memory errors during training
- **Solution**: Reduce batch size or sequence length

## References

- Malhotra, P., et al. (2016). "LSTM-based encoder-decoder for multi-sensor anomaly detection."
- Hundman, K., et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding."
- Research papers on time-series anomaly detection using recurrent neural networks

---

⭐ If you find this project useful, please consider giving it a star!
