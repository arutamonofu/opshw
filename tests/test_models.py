import numpy as np
import pytest

from opshw.metrics import mse
from opshw.models import LinearRegression


def test_fit_analytically(
    regression_model: LinearRegression,
    sample_linear_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_linear_data
    regression_model.fit_analytically(X, y)

    assert regression_model.theta is not None
    # y = 1 + 2*x -> theta = [1.0, 2.0]
    expected_theta = np.array([1.0, 2.0])
    assert np.allclose(regression_model.theta, expected_theta, atol=1e-5)


def test_fit_numerically(
    regression_model: LinearRegression,
    sample_linear_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_linear_data
    # Стартуем с нулевых параметров, за достаточное число итераций ошибка падает
    regression_model.fit_numerically(X, y, learning_rate=0.01, learning_iterations=500)

    assert regression_model.theta is not None
    predictions = regression_model.predict(X)
    # MSE должно быть близко к 0 после обучения
    assert mse(y, predictions) < 0.1


def test_predict_shape(
    regression_model: LinearRegression,
    sample_linear_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_linear_data
    regression_model.fit_analytically(X, y)

    preds = regression_model.predict(X)
    assert preds.shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("learning_rate", "iterations"),
    [
        (0.01, 300),
        (0.015, 200),
    ],
)
def test_fit_numerically_hyperparameters(
    regression_model: LinearRegression,
    sample_linear_data: tuple[np.ndarray, np.ndarray],
    learning_rate: float,
    iterations: int,
) -> None:
    X, y = sample_linear_data
    regression_model.fit_numerically(X, y, learning_rate, iterations)
    preds = regression_model.predict(X)
    assert mse(y, preds) < 0.5
