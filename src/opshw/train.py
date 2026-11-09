import argparse
import contextlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from opshw.metrics import mse
from opshw.models import LinearRegression


def get_git_metadata() -> dict[str, str]:
    """Извлекает метаданные Git (коммит, ветку, автора) для стандартизации логирования."""
    tags: dict[str, str] = {}
    with contextlib.suppress(subprocess.SubprocessError, FileNotFoundError, OSError):
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        tags["git.commit"] = commit

    with contextlib.suppress(subprocess.SubprocessError, FileNotFoundError, OSError):
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tags["git.branch"] = branch

    with contextlib.suppress(subprocess.SubprocessError, FileNotFoundError, OSError):
        author = subprocess.check_output(
            ["git", "log", "-1", "--format=%an <%ae>"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tags["git.author"] = author

    return tags


def train_model(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    learning_rate: float = 0.01,
    learning_iterations: int = 50,
    method: str = "numerical",
    experiment_name: str = "linear_regression",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    save_artifacts: bool = True,
) -> tuple[LinearRegression, str]:
    """Обучает LinearRegression с централизованным трекингом в MLflow."""
    if X is None:
        X = np.array([[1.0], [3.0]])
    if y is None:
        y = np.array([2.0, 4.0])

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    model = LinearRegression()

    with mlflow.start_run(run_name=run_name) as run:
        # Логирование тегов (контракт команды и воспроизводимость)
        git_tags = get_git_metadata()
        if git_tags:
            mlflow.set_tags(git_tags)

        # Логирование гиперпараметров
        mlflow.log_params(
            {
                "method": method,
                "learning_rate": learning_rate,
                "learning_iterations": learning_iterations,
                "n_samples": int(X.shape[0]),
            }
        )

        if method == "numerical":

            def on_epoch(epoch: int, loss: float) -> None:
                mlflow.log_metric("loss", loss, step=epoch)

            model.fit_numerically(
                X,
                y,
                learning_rate=learning_rate,
                learning_iterations=learning_iterations,
                callback=on_epoch,
            )
            final_loss = (
                model.loss_history[-1]
                if model.loss_history
                else float(mse(y, model.predict(X)))
            )
        elif method == "analytical":
            model.fit_analytically(X, y)
            final_loss = (
                model.loss_history[-1]
                if model.loss_history
                else float(mse(y, model.predict(X)))
            )
        else:
            raise ValueError(f"Unknown training method: {method}")

        mlflow.log_metric("final_mse", final_loss)

        # Логирование артефактов
        if save_artifacts and model.theta is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Артефакт 1: Веса модели и параметры в JSON
                weights_file = tmp_path / "model_weights.json"
                weights_data = {
                    "method": method,
                    "theta": model.theta.tolist(),
                    "final_mse": final_loss,
                    "learning_rate": learning_rate,
                    "learning_iterations": learning_iterations,
                }
                weights_file.write_text(json.dumps(weights_data, indent=2))
                mlflow.log_artifact(str(weights_file), artifact_path="model")

                # Артефакт 2: График динамики функции потерь
                if method == "numerical" and model.loss_history:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(
                        range(len(model.loss_history)),
                        model.loss_history,
                        label="MSE loss",
                        color="#1f77b4",
                    )
                    ax.set_title(
                        f"Convergence Curve (lr={learning_rate}, iters={learning_iterations})"
                    )
                    ax.set_xlabel("Epoch / Iteration")
                    ax.set_ylabel("Loss (MSE)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend()
                    plot_file = tmp_path / "loss_curve.png"
                    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    mlflow.log_artifact(str(plot_file), artifact_path="plots")

        run_id = run.info.run_id

    return model, run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Linear Regression model with MLflow experiment tracking."
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for numerical optimization (default: 0.01)",
    )
    parser.add_argument(
        "--iters",
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations (default: 50)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["numerical", "analytical"],
        default="numerical",
        help="Optimization method (default: numerical)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="linear_regression",
        help="MLflow experiment name (default: linear_regression)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (optional)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db or local mlruns)",
    )

    args = parser.parse_args()

    model, run_id = train_model(
        learning_rate=args.lr,
        learning_iterations=args.iters,
        method=args.method,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
    )

    print("Training completed successfully!")
    print(f"MLflow Run ID: {run_id}")
    print(f"Learned parameters (theta): {model.theta}")
    print(f"Loss history length: {len(model.loss_history)}")


if __name__ == "__main__":
    main()
