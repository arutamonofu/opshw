import os

import pytest
import torch
from torch import nn

from opshw.debugging import (
    check_parameters_and_gradients,
    check_tensor_finite,
    compute_gradient_norms,
    debug_breakpoint,
    gradient_anomaly_context,
    run_gradcheck,
)
from opshw.torch_model import DeepMLP


def test_deep_mlp_forward_and_shapes() -> None:
    model = DeepMLP(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        num_layers=3,
        activation="relu",
        use_batch_norm=True,
    )
    x = torch.randn(8, 2)
    out = model(x)
    assert out.shape == (8, 1)


def test_compute_gradient_norms() -> None:
    model = DeepMLP(
        input_dim=1, hidden_dim=10, output_dim=1, num_layers=2, activation="relu"
    )
    x = torch.randn(4, 1)
    target = torch.randn(4, 1)
    criterion = nn.MSELoss()

    loss = criterion(model(x), target)
    loss.backward()

    norms = compute_gradient_norms(model)
    assert len(norms) > 0
    for norm_val in norms.values():
        assert isinstance(norm_val, float)
        assert norm_val >= 0.0


def test_sigmoid_gradient_vanishing() -> None:
    torch.manual_seed(42)
    model = DeepMLP(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_layers=6,
        activation="sigmoid",
    )
    x = torch.randn(16, 1)
    target = torch.randn(16, 1)
    criterion = nn.MSELoss()

    loss = criterion(model(x), target)
    loss.backward()

    norms = compute_gradient_norms(model)
    first_layer_norm = norms["network.0.weight"]
    last_layer_norm = norms["network.12.weight"]

    # In deep sigmoid networks without special init, earlier layers suffer from vanishing gradients
    assert first_layer_norm < last_layer_norm


def test_check_tensor_finite_and_anomaly() -> None:
    finite_tensor = torch.tensor([1.0, 2.0, 3.0])
    assert check_tensor_finite(finite_tensor, "finite") is True

    nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(FloatingPointError, match="contains NaN=True"):
        check_tensor_finite(nan_tensor, "nan_test")

    inf_tensor = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(FloatingPointError, match="Inf=True"):
        check_tensor_finite(inf_tensor, "inf_test")


def test_check_parameters_and_gradients() -> None:
    model = nn.Linear(2, 1)
    x = torch.randn(4, 2)
    out = model(x).sum()
    out.backward()

    results = check_parameters_and_gradients(model)
    assert results["weight"] is True
    assert results["bias"] is True


def test_gradient_anomaly_context() -> None:
    with gradient_anomaly_context():
        x = torch.randn(4, 1, requires_grad=True)
        y = (x**2).sum()
        y.backward()
        assert x.grad is not None


def test_run_gradcheck() -> None:
    inputs = (torch.randn(3, 2, dtype=torch.float64, requires_grad=True),)

    def simple_func(t: torch.Tensor) -> torch.Tensor:
        return 2.0 * t**2 + 3.0 * t

    assert run_gradcheck(simple_func, inputs) is True


def test_debug_breakpoint_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONBREAKPOINT", "0")
    # Should execute without raising or blocking
    debug_breakpoint("Safe CI breakpoint")
    assert os.getenv("PYTHONBREAKPOINT") == "0"
