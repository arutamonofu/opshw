import argparse
from pathlib import Path

import numpy as np
import yaml


def prepare_data(config_path: str = "params.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    prep_params = params.get("prepare", {})
    n_samples = int(prep_params.get("n_samples", 200))
    noise = float(prep_params.get("noise", 0.1))
    seed = int(prep_params.get("seed", 42))
    test_size = float(prep_params.get("test_size", 0.2))

    rng = np.random.default_rng(seed)
    X = rng.uniform(-5.0, 5.0, size=(n_samples, 1))
    true_slope = 2.5
    true_bias = 1.0
    y = true_slope * X.ravel() + true_bias + rng.normal(0, noise, size=n_samples)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    out_dir = Path("data/prepared")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save train and test datasets
    train_data = np.column_stack((X[train_idx], y[train_idx]))
    test_data = np.column_stack((X[test_idx], y[test_idx]))

    np.savetxt(
        out_dir / "train.csv",
        train_data,
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        out_dir / "test.csv",
        test_data,
        delimiter=",",
        header="x,y",
        comments="",
    )

    print(
        f"Data prepared successfully: {len(train_idx)} train samples, {len(test_idx)} test samples."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data stage for DVC pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to configuration parameters file",
    )
    args = parser.parse_args()
    prepare_data(args.config)


if __name__ == "__main__":
    main()
