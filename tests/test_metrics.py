import numpy as np
import pytest

from opshw.metrics import mse


def test_mse_zero_for_identical_arrays() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (np.array([1.0, 2.0]), np.array([2.0, 3.0]), 1.0),
        (np.array([0.0, 0.0]), np.array([2.0, 4.0]), 10.0),
        (np.array([5.0]), np.array([2.0]), 9.0),
    ],
)
def test_mse_calculation(
    y_true: np.ndarray, y_pred: np.ndarray, expected: float
) -> None:
    assert np.isclose(mse(y_true, y_pred), expected)
