import numpy as np
import pytest

import opshw.data
from opshw.data import download_dataset, load_and_preprocess


def test_download_dataset_raises_connection_error() -> None:
    """Проверка, что без мока обращение к внешнему ресурсу вызывает ConnectionError."""
    with pytest.raises(ConnectionError):
        download_dataset("https://example.com/dataset.csv")


def test_load_and_preprocess_with_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверка работы с внешними зависимостями через monkeypatch."""

    def fake_download(_url: str) -> np.ndarray:
        return np.array([4, 5, 3])

    monkeypatch.setattr(opshw.data, "download_dataset", fake_download)

    result = load_and_preprocess("https://example.com/dataset.csv")

    assert np.array_equal(result, np.array([4.0, 5.0, 3.0]))
    assert result.dtype == float


def test_monkeypatch_isolation() -> None:
    """Проверка, что monkeypatch действует изолированно и не влияет на последующие тесты."""
    with pytest.raises(ConnectionError):
        load_and_preprocess("https://example.com/dataset.csv")
