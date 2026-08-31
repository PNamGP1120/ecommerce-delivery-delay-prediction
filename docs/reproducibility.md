# Reproducibility Guide

## Environment

The project uses:

```text
Python 3.14
uv
pyproject.toml
uv.lock
```

Create/sync the environment:

```bash
uv sync
```

---

## Full Validation

```bash
pytest -q
```

Current verified suite:

```text
23 passed
```

---

## Phase 3 — Feature Engineering

```bash
python -m src.features.build_features
```

Expected key output:

```text
features.parquet: 96,450 rows, 44 columns
train:      67,515
validation: 14,467
test:       14,468
```

---

## Phase 4A — Baselines

```bash
python -m src.models.train
```

Produces baseline model-comparison artifacts.

---

## Phase 4B — Temporal Tuning

Smoke test:

```bash
python -m src.models.tune --quick
```

Full search:

```bash
python -m src.models.tune
```

Full search:

```text
22 configurations
x 4 temporal folds
= 88 model fits
```

Selected candidate:

```text
xgboost_03
```

---

## Phase 5 — Evaluation

```bash
python -m src.models.analyze
```

This exports:

```text
ranking metrics
threshold diagnostics
calibration
temporal robustness
segment analysis
feature drift
native importance
permutation importance
SHAP
```

---

## Phase 6 — Deployment Metadata

```bash
python -m src.serving.build_deployment_config
```

Creates:

```text
models/deployment_config.json
```

---

## Run API Locally

```bash
fastapi dev app/main.py
```

or:

```bash
uv run uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

Build:

```bash
docker build -t delivery-delay-api .
```

Run:

```bash
docker run --rm \
  -p 8000:8000 \
  delivery-delay-api
```

Docker Compose:

```bash
docker compose up --build
```

---

## Recommended Reproduction Order

```bash
uv sync

python -m src.features.build_features
pytest tests/test_features.py -q

python -m src.models.train
pytest tests/test_models.py -q

python -m src.models.tune
pytest tests/test_temporal_tuning.py -q

python -m src.models.analyze
pytest tests/test_model_analysis.py -q

python -m src.serving.build_deployment_config
pytest tests/test_serving.py tests/test_api.py -q

pytest -q
```

---

## Important Evaluation Caveat

Do not use the original test diagnostic to make additional feature-selection or hyperparameter decisions.

The test period was already observed during Phase 4A and influenced the choice to perform Phase 4B.

Further experimentation should use new chronological development windows or a truly unseen future period.
