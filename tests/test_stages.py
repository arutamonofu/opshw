import json
from pathlib import Path

import numpy as np
import yaml

from opshw.stages.evaluate import evaluate_stage
from opshw.stages.prepare import prepare_data
from opshw.stages.train import calculate_file_hash, train_stage


def test_calculate_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.txt"
    test_file.write_text("mlops-basics", encoding="utf-8")
    hash_val = calculate_file_hash(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 32


def test_prepare_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_content = {
        "prepare": {
            "n_samples": 50,
            "noise": 0.05,
            "seed": 123,
            "test_size": 0.2,
        }
    }
    config_file = tmp_path / "params.yaml"
    config_file.write_text(yaml.dump(config_content), encoding="utf-8")

    prepare_data(str(config_file))

    train_csv = tmp_path / "data/prepared/train.csv"
    test_csv = tmp_path / "data/prepared/test.csv"

    assert train_csv.exists()
    assert test_csv.exists()

    train_data = np.loadtxt(train_csv, delimiter=",", skiprows=1)
    test_data = np.loadtxt(test_csv, delimiter=",", skiprows=1)

    assert len(train_data) == 40
    assert len(test_data) == 10
    assert train_data.shape[1] == 2


def test_train_and_evaluate_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_content = {
        "prepare": {
            "n_samples": 40,
            "noise": 0.05,
            "seed": 42,
            "test_size": 0.25,
        },
        "train": {
            "learning_rate": 0.001,
            "iterations": 20,
            "method": "numerical",
            "experiment_name": "test_pipeline",
            "run_name": "test_run",
        },
        "evaluate": {
            "metrics_file": "metrics.json",
        },
    }
    config_file = tmp_path / "params.yaml"
    config_file.write_text(yaml.dump(config_content), encoding="utf-8")

    # Step 1: prepare
    prepare_data(str(config_file))

    # Step 2: train
    train_stage(config_path=str(config_file), data_path="data/prepared/train.csv")

    model_file = tmp_path / "models/model.json"
    assert model_file.exists()

    with open(model_file, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    assert "theta" in model_meta
    assert len(model_meta["theta"]) == 2
    assert "data_hash" in model_meta
    assert "mlflow_run_id" in model_meta

    # Step 3: evaluate
    evaluate_stage(
        config_path=str(config_file),
        data_path="data/prepared/test.csv",
        model_path="models/model.json",
    )

    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "test_mse" in metrics
    assert "test_mae" in metrics
    assert "test_r2" in metrics
    assert isinstance(metrics["test_mse"], float)
