from collections.abc import Callable

import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        self.theta = None
        self.loss_history: list[float] = []

    def fit_numerically(
        self,
        X,
        y,
        learning_rate: float,
        learning_iterations: int,
        callback: Callable[[int, float], None] | None = None,
    ) -> None:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.zeros(X_extended.shape[1])
        self.loss_history = []

        for i in range(learning_iterations):
            # Вектор ошибок
            e = X_extended @ self.theta - y

            # MSE
            mse_val = float(np.mean(e**2))
            self.loss_history.append(mse_val)
            if callback is not None:
                callback(i, mse_val)

            # Градиент
            mse_gradient = X_extended.T @ e
            mse_antigradient = -mse_gradient

            # Обновлённые параметры
            self.theta += learning_rate * mse_antigradient

    def fit_analytically(self, X, y) -> None:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))

        A = X_extended.T @ X_extended
        b = X_extended.T @ y
        A_inv = np.linalg.pinv(A)

        self.theta = A_inv @ b
        e = X_extended @ self.theta - y
        self.loss_history = [float(np.mean(e**2))]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))
        return X_extended @ self.theta
