# Phase 6 — Productionization & Deployment

## Serving boundary

The API accepts the finalized 41-feature order-level vector produced at
`order_approved_at`.

It intentionally does **not** perform raw Olist joins online.

The production boundary is:

```text
source systems
    ↓
feature pipeline / feature service
    ↓
41-feature contract
    ↓
FastAPI inference service
    ↓
risk_score + risk_band + requires_review
```

This keeps the serving layer deterministic and prevents expensive item,
payment, product, seller and geolocation joins from being executed for every
prediction request.

## Risk score semantics

`risk_score` is the positive-class model output used for ranking delivery risk.

Phase 5 showed poor probability calibration and temporal threshold instability,
so the API explicitly returns:

```json
"calibrated_probability": false
```

Do not describe a score such as `0.60` as a 60% probability of late delivery.

Risk bands are defined from temporal OOF score quantiles.

The default `requires_review` threshold is copied from the Phase 4B
development-OOF operating threshold with Recall >= 50%. It remains an
operational policy and must be monitored under drift.

## Dependencies

Add the serving dependency:

```bash
uv add "fastapi[standard]"
```

## Build deployment metadata

Before starting the API or building the Docker image:

```bash
python -m src.serving.build_deployment_config
```

This creates:

```text
models/deployment_config.json
```

It contains:

- model candidate and SHA256;
- 41-feature contract count;
- OOF score quantiles;
- OOF-based risk bands;
- development-selected action threshold;
- monitoring reference values;
- explicit calibration warning.

## Run locally

```bash
fastapi dev app/main.py
```

or:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### GET /health

Confirms the process loaded the model.

### GET /model-info

Returns candidate, model version, feature count, score semantics and action
policy.

### POST /predict

Scores one order feature vector.

### POST /predict/batch

Scores a batch of feature vectors.

### GET /monitoring/snapshot

Returns in-process operational telemetry and development reference statistics.

## Example response

```json
{
  "order_id": "abc123",
  "request_id": "c5c6...",
  "risk_score": 0.61,
  "risk_band": "high",
  "requires_review": true,
  "action_threshold": 0.525952,
  "calibrated_probability": false,
  "model_version": "xgboost_03:..."
}
```

## Environment variables

```text
APP_NAME
APP_VERSION
MODEL_PATH
MODEL_METADATA_PATH
DEPLOYMENT_CONFIG_PATH
LOG_LEVEL
MAX_BATCH_SIZE
RISK_ACTION_THRESHOLD
```

`RISK_ACTION_THRESHOLD` is optional and allows the operational threshold to be
changed without retraining the model. Changing it changes business policy, not
model ranking quality.

## Tests

```bash
pytest -q
```

## Docker

The Dockerfile follows the current FastAPI container recommendation of building
from an official Python image rather than the deprecated
`tiangolo/uvicorn-gunicorn-fastapi` image.

It also uses the official uv container binary and the locked project
environment.

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

Or:

```bash
docker compose up --build
```

## Monitoring interpretation

The built-in endpoint is a lightweight portfolio monitoring hook, not a
replacement for Prometheus/OpenTelemetry in a larger production platform.

At minimum monitor:

- request latency and failures;
- score mean/distribution;
- review rate;
- risk-band distribution;
- missing feature values;
- upstream feature freshness;
- later realized late-delivery rate;
- calibration and PR-AUC on matured labels;
- PSI for `promised_delivery_days` and other important features.

Phase 5 identified `promised_delivery_days` as both highly predictive and
strongly shifted, so it deserves a dedicated drift alert.

## Production limitations

- The Olist dataset is historical.
- The observed test period is not a pristine untouched final holdout.
- Model probabilities are not calibrated.
- A fixed score threshold was not temporally stable.
- The API assumes an upstream service can reproduce the exact Phase 3 feature
  contract at approval time.
- Runtime telemetry is in memory and resets when the process restarts.

For a real multi-instance deployment, export metrics/logs to an external
monitoring backend and version the feature pipeline together with the model.
