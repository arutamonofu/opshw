# opshw — MLOps Basics

Учебный репозиторий по дисциплине MLOps Basics: DVC-пайплайны, трекинг в MLflow, инфраструктура на Docker Compose, отладка и профилирование моделей на PyTorch.

## Установка и запуск тестов

```bash
uv sync --all-groups
uv run ruff check
uv run pytest
```

## Инфраструктура (Docker Compose)

Запуск MinIO (S3 хранилище) и MLflow Tracking Server:

```bash
docker compose up -d
```

- MinIO Console: http://localhost:9001 (minioadmin / minioadmin)
- MLflow Tracking: http://localhost:5000
- Переменные окружения: см. `.env.example`

## DVC Пайплайн и эксперименты

Пайплайн описан в `dvc.yaml` и конфигурируется через `params.yaml`:
- `prepare`: подготовка train/test выборок в `data/prepared/`.
- `train`: обучение модели, сохранение в `models/model.json`, логирование в MLflow (с сохранением `git.commit` и `dvc.data_hash`).
- `evaluate`: расчет метрик в `metrics.json`.

```bash
# Воспроизведение пайплайна
uv run dvc repro

# Запуск изолированного эксперимента с новыми гиперпараметрами
uv run dvc exp run -S train.learning_rate=0.002

# Сравнение экспериментов
uv run dvc exp show

# Применение лучшего эксперимента
uv run dvc exp apply <exp_id>

# Синхронизация данных с MinIO S3
uv run dvc push
uv run dvc pull
```

## Отладка PyTorch (Debugging)

Инструменты отладки реализованы в `opshw.debugging`:
- Расчет норм градиентов по слоям (анализ затухания градиентов в Sigmoid vs ReLU).
- Детекция аномалий градиентов (`torch.autograd.set_detect_anomaly`).
- Верификация градиентов через `torch.autograd.gradcheck`.
- Точки останова `breakpoint()` с возможностью отключения через `export PYTHONBREAKPOINT=0`.

Запуск демонстрации отладки:
```bash
uv run opshw-debug
```

## Профилирование PyTorch (Profiler & Perfetto)

Изолированный Debug Runner на базе `torch.profiler` (`opshw.profiler`):
- Расписание сбора метрик: `wait=1`, `warmup=1`, `active=3`, вызов `prof.step()`.
- Профилирование процессорного времени (`CPU total`) и выявление узкого места загрузки батчей DataLoader.
- Профилирование памяти тензоров (`profile_memory=True`).
- Экспорт трассировки событий в Chrome Trace JSON для анализа в Perfetto UI (https://ui.perfetto.dev) или `chrome://tracing`.

Запуск профилировщика:
```bash
uv run opshw-profile --output-trace artifacts/profiler_trace.json
```
