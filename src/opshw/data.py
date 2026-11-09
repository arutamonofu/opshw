import numpy as np


def download_dataset(url: str) -> np.ndarray:
    """Загрузка данных из внешнего источника."""
    # Заглушка реального сетевого запроса к удаленному источнику
    raise ConnectionError(f"Failed to connect to {url}")


def load_and_preprocess(url: str) -> np.ndarray:
    """Загружает данные и приводит к типу float."""
    raw = download_dataset(url)
    return np.asarray(raw, dtype=float)
