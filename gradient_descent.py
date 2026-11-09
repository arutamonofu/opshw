import numpy as np

from opshw import LinearRegression

X = np.array([[1], [3]])
y = np.array([2, 4])

model = LinearRegression()
model.fit_numerically(X, y, 0.01, 50)
print("Параметры, вычисленные численным методом:", model.theta)

model.fit_analytically(X, y)
print("Параметры, вычисленные аналитическим методом:", model.theta)
