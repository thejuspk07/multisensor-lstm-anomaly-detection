import numpy as np


def compute_reconstruction_error(X, X_pred):
    return np.mean((X - X_pred) ** 2, axis=(1, 2))


def detect_anomalous_cycles(errors, threshold):
    return np.where(errors > threshold)[0]


def estimate_rul(current_error, error_min, error_max, max_rul=130):
    norm_error = (current_error - error_min) / (error_max - error_min + 1e-8)
    norm_error = np.clip(norm_error, 0, 1)
    health_index = 1 - norm_error
    rul = health_index * max_rul
    return health_index, rul
