import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, TensorDataset

from opshw.torch_model import DeepMLP


@dataclass
class ProfilerConfig:
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    profile_memory: bool = True
    record_shapes: bool = True
    trace_output_path: str = "artifacts/profiler_trace.json"
    batch_size: int = 32
    simulate_slow_loader: bool = True
    slow_loader_delay: float = (
        0.005  # задержка для демонстрации bottleneck загрузки батча
    )


def build_synthetic_dataloader(
    n_samples: int = 512, batch_size: int = 32
) -> DataLoader:
    """Создает синтетический DataLoader для воспроизводимого профилирования."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, 8)
    y = torch.randn(n_samples, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_debug_profiler(
    config: ProfilerConfig | None = None,
) -> tuple[str, str, Path]:
    """Запускает цикл профилирования PyTorch с расписанием шагов, сбором метрик памяти и экспортом в Chrome Trace."""
    if config is None:
        config = ProfilerConfig()

    total_steps = (
        config.wait_steps + config.warmup_steps + config.active_steps
    ) * config.repeat

    dataloader = build_synthetic_dataloader(batch_size=config.batch_size)
    data_iter = iter(dataloader)

    model = DeepMLP(
        input_dim=8,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        activation="relu",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Настройка расписания профилировщика: wait -> warmup -> active -> repeat
    prof_schedule = schedule(
        wait=config.wait_steps,
        warmup=config.warmup_steps,
        active=config.active_steps,
        repeat=config.repeat,
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_file = Path(config.trace_output_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            # Шаг 1: Симуляция загрузки батча (load_batch bottleneck)
            with record_function("## load_batch ##"):
                if config.simulate_slow_loader and config.slow_loader_delay > 0:
                    time.sleep(config.slow_loader_delay)
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)

            # Шаг 2: Прямой проход (forward pass)
            with record_function("## model_forward ##"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Шаг 3: Обратный проход (backward pass)
            with record_function("## model_backward ##"):
                optimizer.zero_grad()
                loss.backward()

            # Шаг 4: Обновление весов (optimizer step)
            with record_function("## optimizer_step ##"):
                optimizer.step()

            # Явный сигнал профайлеру о завершении шага итерации
            prof.step()

    # Экспорт трассировки для Chrome Tracing / Perfetto
    prof.export_chrome_trace(str(trace_file))

    # Формирование отчетов по CPU времени и потреблению памяти
    cpu_time_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
    memory_table = prof.key_averages().table(
        sort_by="self_cpu_memory_usage", row_limit=10
    )

    return cpu_time_table, memory_table, trace_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler and Debug Runner with Chrome Trace / Perfetto export."
    )
    parser.add_argument(
        "--output-trace",
        type=str,
        default="artifacts/profiler_trace.json",
        help="Path to export Chrome Trace JSON (default: artifacts/profiler_trace.json)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=1,
        help="Schedule wait steps (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Schedule warmup steps (default: 1)",
    )
    parser.add_argument(
        "--active",
        type=int,
        default=3,
        help="Schedule active steps (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--simulate-bottleneck",
        action="store_true",
        default=True,
        help="Simulate I/O DataLoader bottleneck (default: True)",
    )

    args = parser.parse_args()

    cfg = ProfilerConfig(
        wait_steps=args.wait,
        warmup_steps=args.warmup,
        active_steps=args.active,
        batch_size=args.batch_size,
        trace_output_path=args.output_trace,
        simulate_slow_loader=args.simulate_bottleneck,
    )

    print("=" * 70)
    print("Запуск PyTorch Profiler (Debug Runner)")
    print(
        f"Расписание: wait={cfg.wait_steps}, warmup={cfg.warmup_steps}, active={cfg.active_steps}"
    )
    print("=" * 70)

    cpu_table, mem_table, trace_path = run_debug_profiler(cfg)

    print("\n--- Топ операций по общему процессорному времени (CPU Total Time) ---")
    print(cpu_table)

    print("\n--- Анализ потребления памяти тензорами (Memory Summary) ---")
    print(mem_table)

    print("=" * 70)
    print(f"✅ Трассировка успешно экспортирована: {trace_path}")
    print("Для визуализации откройте:")
    print("  1. Perfetto UI: https://ui.perfetto.dev (нажмите 'Open trace file')")
    print("  2. Либо Chrome Tracing: chrome://tracing в браузере Google Chrome")
    print("=" * 70)


if __name__ == "__main__":
    main()
