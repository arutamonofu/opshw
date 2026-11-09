import argparse
import os
from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
from torch import nn

from opshw.torch_model import DeepMLP


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


def run_debugging_demo() -> None:
    """Демонстрирует инструменты отладки: нормы градиентов, anomaly detection, gradcheck и pdb."""
    print("=" * 60)
    print("1. Анализ норм градиентов: Затухание в Sigmoid vs ReLU")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(32, 1)
    target = torch.randn(32, 1)
    criterion = nn.MSELoss()

    # Модель с Sigmoid (демонстрация затухания градиента)
    sigmoid_model = DeepMLP(
        input_dim=1, hidden_dim=32, num_layers=5, activation="sigmoid"
    )
    out_sig = sigmoid_model(x)
    loss_sig = criterion(out_sig, target)
    loss_sig.backward()
    sig_norms = compute_gradient_norms(sigmoid_model)

    print("Нормы градиентов для Sigmoid сети (затухание к первым слоям):")
    for name, norm in sig_norms.items():
        if "weight" in name:
            print(f"  {name:30s} -> norm: {norm:.6e}")

    # Модель с ReLU (сохранение величины градиентов)
    relu_model = DeepMLP(input_dim=1, hidden_dim=32, num_layers=5, activation="relu")
    out_relu = relu_model(x)
    loss_relu = criterion(out_relu, target)
    loss_relu.backward()
    relu_norms = compute_gradient_norms(relu_model)

    print("\nНормы градиентов для ReLU сети:")
    for name, norm in relu_norms.items():
        if "weight" in name:
            print(f"  {name:30s} -> norm: {norm:.6e}")

    print("\n" + "=" * 60)
    print("2. Проверка аномалий градиентов (Gradient Anomaly Detection)")
    print("=" * 60)
    with gradient_anomaly_context():
        test_in = torch.randn(8, 1, requires_grad=True)
        test_out = torch.sin(test_in).sum()
        test_out.backward()
        print("Anomaly detection context: Градиенты вычислены корректно.")

    print("\n" + "=" * 60)
    print("3. Верификация градиентов через torch.autograd.gradcheck")
    print("=" * 60)
    # Для gradcheck требуются double (float64) тензоры
    input_double = (torch.randn(4, 2, dtype=torch.float64, requires_grad=True),)

    def custom_func(t: torch.Tensor) -> torch.Tensor:
        return torch.pow(t, 3) + 2.0 * t

    gradcheck_passed = run_gradcheck(custom_func, input_double)
    print(f"Gradcheck успешно пройден: {gradcheck_passed}")

    print("\n" + "=" * 60)
    print("4. Python Debugger (PDB) и PYTHONBREAKPOINT=0")
    print("=" * 60)
    print("Проверка вызова debug_breakpoint с PYTHONBREAKPOINT=0...")
    os.environ["PYTHONBREAKPOINT"] = "0"
    debug_breakpoint("Проверка пропуска брейкпоинта в автоматическом режиме")
    print(
        "Точка останова успешно пропущена благодаря PYTHONBREAKPOINT=0 (CI/CD режим)."
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch debugging and gradient analysis demo."
    )
    parser.parse_args()
    run_debugging_demo()


if __name__ == "__main__":
    main()
