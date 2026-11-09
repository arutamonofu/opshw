import numpy as np

from opshw import train_model

X = np.array([[1.0], [3.0]])
y = np.array([2.0, 4.0])

model_num, run_id_num = train_model(
    X, y, learning_rate=0.01, learning_iterations=50, method="numerical"
)
print("Параметры, вычисленные численным методом:", model_num.theta)
print(f"MLflow Run ID: {run_id_num}")

model_ana, run_id_ana = train_model(X, y, method="analytical")
print("Параметры, вычисленные аналитическим методом:", model_ana.theta)
print(f"MLflow Run ID: {run_id_ana}")
