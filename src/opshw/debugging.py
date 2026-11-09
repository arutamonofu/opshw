import os
from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
from torch import nn


def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Вычисляет L2-норму градиентов для каждого параметрического слоя модели.

    Позволяет диагностировать затухание (vanishing) или взрыв (exploding)
    градиентов по слоям сети после backward() шага.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm_val = float(torch.linalg.vector_norm(param.grad).item())
            norms[name] = norm_val
        else:
            norms[name] = 0.0
    return norms


def check_tensor_finite(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Проверяет отсутствие NaN и Inf значений в тензоре."""
    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()
    if is_nan or is_inf:
        raise FloatingPointError(
            f"Anomaly detected in {name}: contains NaN={is_nan}, Inf={is_inf}"
        )
    return True


def check_parameters_and_gradients(model: nn.Module) -> dict[str, bool]:
    """Проверяет все параметры и градиенты модели на конечность (отсутствие NaN/Inf)."""
    results: dict[str, bool] = {}
    for name, param in model.named_parameters():
        check_tensor_finite(param.data, f"parameter {name}")
        if param.grad is not None:
            check_tensor_finite(param.grad.data, f"gradient {name}")
        results[name] = True
    return results


def gradient_anomaly_context() -> AbstractContextManager:
    """Контекстный менеджер PyTorch для отслеживания источника некорректных градиентов (NaN/Inf).

    Использует torch.autograd.set_detect_anomaly(True).
    """
    return torch.autograd.set_detect_anomaly(True)


def run_gradcheck(
    func: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    eps: float = 1e-6,
    atol: float = 1e-4,
) -> bool:
    """Верифицирует аналитические градиенты с помощью конечных разностей через torch.autograd.gradcheck."""
    return torch.autograd.gradcheck(
        func, inputs, eps=eps, atol=atol, raise_exception=True
    )


def debug_breakpoint(hint: str = "") -> None:
    """Безопасная точка прерывания pdb.

    Если задана переменная окружения PYTHONBREAKPOINT=0, брейкпоинт пропускается без остановки программы.
    """
    if os.getenv("PYTHONBREAKPOINT") == "0":
        return

    if hint:
        print(f"\n[DEBUG BREAKPOINT]: {hint}")
        print(
            "PDB commands: 'n' (next), 's' (step), 'p <var>' (print), 'c' (continue), 'q' (quit)\n"
        )
    breakpoint()  # noqa: T100
