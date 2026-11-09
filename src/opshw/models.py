import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        self.theta = None

    def fit_numerically(self, X, y, learning_rate, learning_iterations) -> None:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.zeros(X_extended.shape[1])

        for i in range(learning_iterations):
            # Вектор ошибок
            e = X_extended @ self.theta - y

            # MSE
            _mse = np.mean(e**2)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))
        return X_extended @ self.theta
