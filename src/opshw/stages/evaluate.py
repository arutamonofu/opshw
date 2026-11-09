import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import yaml

from opshw.metrics import mse
from opshw.models import LinearRegression


def evaluate_stage(
    config_path: str = "params.yaml",
    data_path: str = "data/prepared/test.csv",
    model_path: str = "models/model.json",
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    eval_params = params.get("evaluate", {})
    metrics_file = Path(eval_params.get("metrics_file", "metrics.json"))

    # Load model parameters
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    model = LinearRegression()
    model.theta = np.array(model_data["theta"])

    # Load test data
    test_file = Path(data_path)
    if not test_file.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    raw_test = np.loadtxt(test_file, delimiter=",", skiprows=1)
    X_test = raw_test[:, :-1]
    y_test = raw_test[:, -1]

    # Predict and evaluate
    y_pred = model.predict(X_test)
    test_mse = float(mse(y_test, y_pred))
    test_mae = float(np.mean(np.abs(y_test - y_pred)))
    total_variance = float(np.sum((y_test - np.mean(y_test)) ** 2))
    residual_variance = float(np.sum((y_test - y_pred) ** 2))
    r2_score = (
        float(1.0 - (residual_variance / total_variance)) if total_variance > 0 else 0.0
    )

    metrics = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": r2_score,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log evaluation metrics to MLflow run
    run_id = model_data.get("mlflow_run_id")
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "eval_mse": test_mse,
                    "eval_mae": test_mae,
                    "eval_r2": r2_score,
                }
            )

    print(f"Evaluation completed. Metrics saved to {metrics_file}: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage for DVC pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to configuration parameters file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/prepared/test.csv",
        help="Path to test data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.json",
        help="Path to trained model artifact",
    )
    args = parser.parse_args()
    evaluate_stage(args.config, args.data, args.model)


if __name__ == "__main__":
    main()
