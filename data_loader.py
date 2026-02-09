import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_11", "sensor_12"
]


def load_main_dataset(path):
    """
    Load combined C-MAPSS dataset
    """
    df = pd.read_csv(path)
    return df


def select_sensors(df):
    return df[SELECTED_SENSORS]


def normalize_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, seq_len):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return np.array(X)


def split_by_engine(df):
    engines = {}
    for eid in df["engine_id"].unique():
        engines[eid] = df[df["engine_id"] == eid]
    return engines
