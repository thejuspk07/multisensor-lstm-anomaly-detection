from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io, base64
from data_loader import split_by_engine
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "CMAPSS_ALL_IN_ONE.csv")
DEMO_MODE = False   # 🔥 Set to False for real (optimized) training


from data_loader import (
    load_main_dataset,
    split_by_engine,
    select_sensors,     
    normalize_data,
    create_sequences
)

from model import build_lstm_autoencoder
from anomaly_detection import (
    compute_reconstruction_error,
    detect_anomalous_cycles,
    estimate_rul
)

app = Flask(__name__)

SEQ_LEN = 30
THRESHOLD = 0.5
MAX_RUL = 130
DATASET_PATH = "datasets/train_FD001.txt"


def train_pipeline_multi_engine(df):
    engines = split_by_engine(df)

    engine_errors = {}
    engine_health = {}

    for eid, engine_df in engines.items():
        sensor_df = select_sensors(engine_df)
        scaled, _ = normalize_data(sensor_df)
        X = create_sequences(scaled, SEQ_LEN)

        if len(X) < 10:
            continue

        model = build_lstm_autoencoder(SEQ_LEN, X.shape[2])
        model.fit(X, X, epochs=3, batch_size=32, verbose=0)

        X_pred = model.predict(X)
        errors = compute_reconstruction_error(X, X_pred)

        engine_errors[eid] = errors
        engine_health[eid] = 1 - np.mean(errors)

    return engine_errors, engine_health



# Global variables to store model results
model = None
engine_errors = None
engine_health = None



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    initialize()

    threshold = float(request.form.get("threshold", 0.5))

    # -------- BAR GRAPH (ENGINE COMPARISON) --------
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(
        engine_health.keys(),
        engine_health.values(),
        color="#1cc88a"
    )
    ax1.set_title("Engine Health Comparison")
    ax1.set_xlabel("Engine ID")
    ax1.set_ylabel("Health Score")

    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    plt.close(fig1)
    buf1.seek(0)
    bar_graph = base64.b64encode(buf1.read()).decode("utf-8")

    # -------- SINGLE ENGINE DETAIL (ENGINE 1) --------
    engine_id = list(engine_errors.keys())[0]
    errors = engine_errors[engine_id]
    cycles = np.arange(len(errors))
    faulty_cycles = detect_anomalous_cycles(errors, threshold)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(cycles, errors, label="Reconstruction Error")
    ax2.axhline(threshold, color="red", linestyle="--")
    ax2.scatter(faulty_cycles, errors[faulty_cycles], color="orange")
    ax2.set_title(f"Engine {engine_id} – Error vs Cycle")

    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    plt.close(fig2)
    buf2.seek(0)
    error_graph = base64.b64encode(buf2.read()).decode("utf-8")

    hi, rul = estimate_rul(errors[-1], errors.min(), errors.max(), MAX_RUL)

    return render_template(
        "dashboard.html",
        bar_graph=bar_graph,
        error_graph=error_graph,
        threshold=threshold,
        health_index=round(float(hi), 3),
        rul=round(float(rul), 2),
        faulty_cycles=faulty_cycles.tolist()
    )

def train_global_model(df):
    sensor_df = select_sensors(df)
    scaled, _ = normalize_data(sensor_df)
    X = create_sequences(scaled, SEQ_LEN)

    model = build_lstm_autoencoder(SEQ_LEN, X.shape[2])
    model.fit(X, X, epochs=3, batch_size=64, verbose=0)

    return model

def compute_engine_errors(model, df):
    engines = split_by_engine(df)
    engine_errors = {}
    engine_health = {}

    for eid, edf in engines.items():
        sensor_df = select_sensors(edf)
        scaled, _ = normalize_data(sensor_df)
        X = create_sequences(scaled, SEQ_LEN)

        if len(X) < SEQ_LEN:
            continue

        X_pred = model.predict(X, verbose=0)
        errors = compute_reconstruction_error(X, X_pred)

        engine_errors[eid] = errors
        engine_health[eid] = 1 - errors.mean()

    return engine_errors, engine_health

def initialize():
    global model, engine_errors, engine_health

    if engine_errors is not None:
        return

    if DEMO_MODE:
        print("⚡ Running in DEMO MODE (no training)")

        # Fake data for UI testing
        engine_errors = {
            1: np.random.rand(150) * 0.4,
            2: np.random.rand(150) * 0.6,
            3: np.random.rand(150) * 0.3,
            4: np.random.rand(150) * 0.5,
        }

        engine_health = {
            eid: 1 - np.mean(err)
            for eid, err in engine_errors.items()
        }
        return

    # REAL TRAINING
    print("🔄 Training global LSTM model...")
    df = load_main_dataset(MAIN_DATASET_PATH)
    model = train_global_model(df)
    engine_errors, engine_health = compute_engine_errors(model, df)
    print("✅ Model ready")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
