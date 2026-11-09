from opshw.data import download_dataset, load_and_preprocess
from opshw.metrics import mse
from opshw.models import LinearRegression


def main() -> None:
    print("Hello from opshw!")


__all__ = [
    "LinearRegression",
    "download_dataset",
    "load_and_preprocess",
    "main",
    "mse",
]
