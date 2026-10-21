import numpy as np

X = np.array([[1], [3]])
y = np.array([2, 4])


class LinearRegression:
    def __init__(self) -> None:
        pass

    def fit_numerically(self, X, y, learning_rate, learning_iterations) -> None:
        X_extended = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.zeros(X_extended.shape[1])

        for i in range(learning_iterations):
            # Вектор ошибок
            e = X_extended @ self.theta - y

            # MSE
            mse = np.mean(e**2)

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
        X_extended = np.column_stack((np.ones(X.shape[0])), X) # type: ignore
        return X_extended @ self.theta


model = LinearRegression()
model.fit_numerically(X, y, 0.1, 50)
print("Параметры, вычисленные численным методом:", model.theta)

model.fit_analytically(X, y)
print("Параметры, вычисленные аналитическим методом:", model.theta)
