import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from opshw.metrics import mse
from opshw.models import LinearRegression

bounded_floats = st.floats(
    min_value=-1e5,
    max_value=1e5,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def array_pairs_same_shape(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=1, max_value=100))
    a = draw(hnp.arrays(dtype=np.float64, shape=(size,), elements=bounded_floats))
    b = draw(hnp.arrays(dtype=np.float64, shape=(size,), elements=bounded_floats))
    return a, b


@st.composite
def regression_data_strategy(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = draw(st.integers(min_value=5, max_value=50))
    n_features = draw(st.integers(min_value=1, max_value=5))
    X = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    y = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_samples,),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return X, y


@st.composite
def synthetic_linear_dataset(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_features = draw(st.integers(min_value=1, max_value=4))
    n_samples = draw(st.integers(min_value=20, max_value=50))
    weights = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_features,),
            elements=st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    bias = draw(
        st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    X = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    y = X @ weights + bias
    return X, y, weights, bias


# --- Properties of MSE ---


@given(array_pairs_same_shape())
def test_mse_non_negative(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Свойство: MSE всегда неотрицательна."""
    y_true, y_pred = pair
    error = mse(y_true, y_pred)
    assert not np.isnan(error)
    assert error >= 0.0


@given(
    hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=bounded_floats,
    )
)
def test_mse_zero_for_identical(y: np.ndarray) -> None:
    """Свойство тождественности: MSE(y, y) равен 0."""
    assert np.isclose(mse(y, y), 0.0, atol=1e-12)


@given(array_pairs_same_shape())
def test_mse_symmetric(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Свойство симметричности: MSE(a, b) == MSE(b, a)."""
    a, b = pair
    assert np.isclose(mse(a, b), mse(b, a), rtol=1e-7, atol=1e-9)


# --- Properties of LinearRegression ---


@given(regression_data_strategy())
def test_predict_shape_and_finite(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Свойство: размерность выхода всегда (N,) и выход не содержит NaN/Inf при валидном входе."""
    X, y = data
    model = LinearRegression()
    model.fit_analytically(X, y)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.all(np.isfinite(preds))


@given(synthetic_linear_dataset())
def test_analytical_fit_reconstructs_linear_function(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    """Свойство: на линейно зависимых данных без шума аналитическое решение точно восстанавливает y."""
    X, y, _weights, _bias = data
    model = LinearRegression()
    model.fit_analytically(X, y)

    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-4)
