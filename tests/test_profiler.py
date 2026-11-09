import json
from pathlib import Path

from opshw.profiler import (
    ProfilerConfig,
    build_synthetic_dataloader,
    run_debug_profiler,
)


def test_build_synthetic_dataloader() -> None:
    loader = build_synthetic_dataloader(n_samples=64, batch_size=16)
    batches = list(loader)
    assert len(batches) == 4
    x, y = batches[0]
    assert x.shape == (16, 8)
    assert y.shape == (16, 1)


def test_run_debug_profiler(tmp_path: Path) -> None:
    trace_file = tmp_path / "test_trace.json"
    cfg = ProfilerConfig(
        wait_steps=1,
        warmup_steps=1,
        active_steps=1,
        repeat=1,
        trace_output_path=str(trace_file),
        simulate_slow_loader=False,
        slow_loader_delay=0.0,
        batch_size=16,
    )

    cpu_table, mem_table, out_path = run_debug_profiler(cfg)

    assert out_path.exists()
    assert str(out_path) == str(trace_file)

    # Verify Chrome Trace JSON format
    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    assert "traceEvents" in trace_data
    assert isinstance(trace_data["traceEvents"], list)
    assert len(trace_data["traceEvents"]) > 0

    # Verify table outputs
    assert "ProfilerStep*" in cpu_table or "CPU total" in cpu_table
    assert "Memory" in mem_table or "Self CPU Mem" in mem_table
