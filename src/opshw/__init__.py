from opshw.data import download_dataset, load_and_preprocess
from opshw.debugging import (
    check_parameters_and_gradients,
    check_tensor_finite,
    compute_gradient_norms,
    debug_breakpoint,
    gradient_anomaly_context,
    run_gradcheck,
)
from opshw.metrics import mse
from opshw.models import LinearRegression
from opshw.torch_model import DeepMLP
from opshw.train import train_model


def main() -> None:
    print("Hello from opshw!")


__all__ = [
    "DeepMLP",
    "LinearRegression",
    "check_parameters_and_gradients",
    "check_tensor_finite",
    "compute_gradient_norms",
    "debug_breakpoint",
    "download_dataset",
    "gradient_anomaly_context",
    "load_and_preprocess",
    "main",
    "mse",
    "run_gradcheck",
    "train_model",
]
