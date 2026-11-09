import numpy as np
import pytest

import opshw.data
from opshw.data import load_and_preprocess
from opshw.metrics import mse
from opshw.models import LinearRegression


@pytest.mark.smoke
def test_e2e_training_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke/E2E тест: сквозной прогон загрузки данных, обучения и оценки."""

    # 1. Загрузка данных через мок внешнего источника
    def fake_download(_url: str) -> np.ndarray:
        # 10 строк, 2 признака + 1 целевая колонка
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 2))
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
        return np.column_stack((X, y))

    monkeypatch.setattr(opshw.data, "download_dataset", fake_download)

    raw_data = load_and_preprocess("https://example.com/mock-dataset.csv")
    assert raw_data.shape == (10, 3)

    X_train = raw_data[:7, :2]
    y_train = raw_data[:7, 2]
    X_val = raw_data[7:, :2]
    y_val = raw_data[7:, 2]

    # 2. Обучение модели
    model = LinearRegression()
    model.fit_analytically(X_train, y_train)

    assert model.theta is not None
    assert len(model.theta) == 3
    assert np.all(np.isfinite(model.theta))

    # 3. Инференс и валидация качества
    predictions = model.predict(X_val)
    val_loss = mse(y_val, predictions)

    assert np.isfinite(val_loss)
    # На линейных данных без шума ошибка близка к 0
    assert val_loss < 1e-4


@pytest.mark.smoke
def test_gradient_descent_loss_decreases() -> None:
    """Smoke тест: численный градиентный спуск уменьшает ошибку и не расходится."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(20, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0

    # Ошибка с весами по умолчанию (нули)
    initial_loss = float(np.mean(y**2))

    model = LinearRegression()
    model.fit_numerically(X, y, learning_rate=0.01, learning_iterations=100)

    assert model.theta is not None
    assert not np.any(np.isnan(model.theta))

    final_loss = mse(y, model.predict(X))
    assert np.isfinite(final_loss)
    # За 100 итераций ошибка должна существенно уменьшиться относительно начальной
    assert final_loss < initial_loss
