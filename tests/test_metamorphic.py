import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from opshw.models import LinearRegression


@pytest.fixture
def regression_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Синтетический датасет для метаморфических тестов."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    true_theta = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    X_ext = np.column_stack((np.ones(50), X))
    y = X_ext @ true_theta + rng.normal(scale=0.1, size=50)
    return X, y


def test_mr_training_sample_permutation_invariance(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """MR1: Инвариантность к перестановке строк обучающей выборки.

    Перестановка порядка объектов в обучающей выборке (X, y)
    не должна менять вычисленные коэффициенты модели.
    """
    X, y = regression_dataset
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(X))

    model_orig = LinearRegression()
    model_orig.fit_analytically(X, y)

    model_perm = LinearRegression()
    model_perm.fit_analytically(X[perm], y[perm])

    assert np.allclose(model_orig.theta, model_perm.theta, atol=1e-8)


def test_mr_target_shift_equivariance(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """MR2: Эквивариантность к сдвигу целевой переменной на константу c.

    Если y_new = y + c, то смещение theta_0 должно увеличиться на c,
    а веса признаков theta_1..D должны остаться неизменными.
    """
    X, y = regression_dataset
    shift_const = 5.25

    model_orig = LinearRegression()
    model_orig.fit_analytically(X, y)

    model_shifted = LinearRegression()
    model_shifted.fit_analytically(X, y + shift_const)

    # theta_0 изменился на shift_const
    assert np.isclose(
        model_shifted.theta[0], model_orig.theta[0] + shift_const, atol=1e-8
    )
    # остальные веса theta_1..D идентичны
    assert np.allclose(model_shifted.theta[1:], model_orig.theta[1:], atol=1e-8)


def test_mr_target_scale_equivariance(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """MR3: Эквивариантность к масштабированию целевой переменной.

    Если y_new = alpha * y, то theta_new = alpha * theta.
    """
    X, y = regression_dataset
    scale = 3.5

    model_orig = LinearRegression()
    model_orig.fit_analytically(X, y)

    model_scaled = LinearRegression()
    model_scaled.fit_analytically(X, scale * y)

    assert np.allclose(model_scaled.theta, scale * model_orig.theta, atol=1e-8)


def test_mr_prediction_permutation_equivariance(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """MR4: Эквивариантность предсказаний к перестановке входных объектов.

    predict(X[perm]) == predict(X)[perm].
    """
    X, y = regression_dataset
    model = LinearRegression()
    model.fit_analytically(X, y)

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(X))

    preds_orig = model.predict(X)
    preds_perm = model.predict(X[perm])

    assert np.allclose(preds_perm, preds_orig[perm], atol=1e-8)


@given(
    shift=st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    data=hnp.arrays(
        dtype=np.float64,
        shape=(30, 3),
        elements=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_mr_hypothesis_linear_transform(
    shift: float, scale: float, data: np.ndarray
) -> None:
    """MR5 (Property + Metamorphic): линейное преобразование предсказаний.

    Для любой обученной модели: predict(X) * scale + shift == predict_with_scaled_shifted_theta(X).
    """
    X = data
    y = np.sum(X, axis=1)

    model = LinearRegression()
    model.fit_analytically(X, y)

    # Если мы масштабируем y и сдвигаем его: y_trans = scale * y + shift
    model_trans = LinearRegression()
    model_trans.fit_analytically(X, scale * y + shift)

    # Предсказания трансформированной модели должны равняться трансформированным предсказаниям
    preds_orig = model.predict(X)
    preds_trans = model_trans.predict(X)

    assert np.allclose(preds_trans, scale * preds_orig + shift, atol=1e-4)
