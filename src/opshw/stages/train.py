import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from opshw.train import get_git_metadata, train_model


def calculate_file_hash(filepath: str | Path) -> str:
    """Вычисляет MD5 хеш файла данных для версионирования и связки с DVC/MLflow."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_stage(
    config_path: str = "params.yaml", data_path: str = "data/prepared/train.csv"
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    learning_rate = float(train_params.get("learning_rate", 0.01))
    iterations = int(train_params.get("iterations", 100))
    method = str(train_params.get("method", "numerical"))
    experiment_name = str(train_params.get("experiment_name", "linear_regression"))
    run_name = str(train_params.get("run_name", "dvc_repro_run"))

    # Load prepared train dataset
    train_file = Path(data_path)
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    raw_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    X = raw_data[:, :-1]
    y = raw_data[:, -1]

    # Calculate data hash for reproducible tracking
    data_hash = calculate_file_hash(train_file)

    # Train model with MLflow experiment tracking
    model, run_id = train_model(
        X=X,
        y=y,
        learning_rate=learning_rate,
        learning_iterations=iterations,
        method=method,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # Log DVC data tracking tags to MLflow
    import mlflow

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("dvc.data_path", str(train_file))
        mlflow.set_tag("dvc.data_hash", data_hash)
        mlflow.set_tag("pipeline.stage", "train")

    # Save model artifact locally for DVC stage tracking
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_output = models_dir / "model.json"

    model_metadata = {
        "theta": model.theta.tolist() if model.theta is not None else [],
        "method": method,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "train_samples": int(X.shape[0]),
        "data_hash": data_hash,
        "mlflow_run_id": run_id,
        "git_tags": get_git_metadata(),
    }

    with open(model_output, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"Training stage complete. Model saved to {model_output}")
    print(f"MLflow Run ID: {run_id}, Data hash: {data_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage for DVC pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to configuration parameters file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/prepared/train.csv",
        help="Path to training data",
    )
    args = parser.parse_args()
    train_stage(args.config, args.data)


if __name__ == "__main__":
    main()
