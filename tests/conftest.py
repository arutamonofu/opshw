import numpy as np
import pytest

from opshw.models import LinearRegression


@pytest.fixture
def sample_linear_data() -> tuple[np.ndarray, np.ndarray]:
    """Синтетические одномерные линейные данные: y = 2x + 1."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    return X, y


@pytest.fixture
def regression_model() -> LinearRegression:
    """Фикстура свежего экземпляра модели."""
    return LinearRegression()
