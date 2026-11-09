from pathlib import Path

import mlflow
import numpy as np
import pytest

from opshw.train import get_git_metadata, train_model


@pytest.fixture
def test_tracking_uri(tmp_path: Path) -> str:
    db_file = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db_file}"


def test_get_git_metadata() -> None:
    tags = get_git_metadata()
    assert isinstance(tags, dict)
    # В репозитории git должны присутствовать ключи коммита и ветки
    assert "git.commit" in tags
    assert "git.branch" in tags


def test_train_model_numerical(test_tracking_uri: str) -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model, run_id = train_model(
        X=X,
        y=y,
        learning_rate=0.01,
        learning_iterations=30,
        method="numerical",
        experiment_name="test_numerical_experiment",
        run_name="test_run_numerical",
        tracking_uri=test_tracking_uri,
        save_artifacts=True,
    )

    assert model.theta is not None
    assert len(model.loss_history) == 30
    assert run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=test_tracking_uri)
    run = client.get_run(run_id)

    # Проверяем параметры
    assert run.data.params["method"] == "numerical"
    assert run.data.params["learning_rate"] == "0.01"
    assert run.data.params["learning_iterations"] == "30"
    assert run.data.params["n_samples"] == "3"

    # Проверяем метрики
    assert "final_mse" in run.data.metrics
    metric_history = client.get_metric_history(run_id, "loss")
    assert len(metric_history) == 30

    # Проверяем наличие артефактов
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "model" in artifact_paths
    assert "plots" in artifact_paths


def test_train_model_analytical(test_tracking_uri: str) -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])

    model, run_id = train_model(
        X=X,
        y=y,
        method="analytical",
        experiment_name="test_analytical_experiment",
        run_name="test_run_analytical",
        tracking_uri=test_tracking_uri,
        save_artifacts=True,
    )

    assert model.theta is not None
    assert np.allclose(model.theta, [1.0, 2.0], atol=1e-4)

    client = mlflow.tracking.MlflowClient(tracking_uri=test_tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["method"] == "analytical"
    assert "final_mse" in run.data.metrics
    assert run.data.metrics["final_mse"] < 1e-6


def test_train_model_invalid_method(test_tracking_uri: str) -> None:
    with pytest.raises(ValueError, match="Unknown training method"):
        train_model(
            method="unsupported_method",
            tracking_uri=test_tracking_uri,
        )
