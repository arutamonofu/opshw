import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_pred - y_true
    return float(np.mean(e**2))
