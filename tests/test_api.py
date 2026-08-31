from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from src.monitoring.telemetry import (
    PredictionTelemetry,
)
from src.serving.model_service import (
    ModelService,
)


class FakeBinaryModel:
    def predict_proba(
        self,
        X,
    ):
        promise = np.asarray(
            X["promised_delivery_days"],
            dtype=float,
        )
        score = np.clip(
            0.8
            - promise / 60.0,
            0.01,
            0.99,
        )
        return np.column_stack(
            [1 - score, score]
        )


def _service():
    config = {
        "model": {
            "candidate": "xgboost_03",
            "family": "xgboost",
            "sha256": "b" * 64,
            "feature_count": 41,
        },
        "risk_score": {
            "semantics": (
                "uncalibrated ranking score"
            ),
            "calibrated_probability": False,
            "risk_band_source": (
                "development OOF quantiles"
            ),
            "bands": {
                "low_max": 0.20,
                "medium_max": 0.40,
                "high_max": 0.60,
            },
            "action_threshold": 0.50,
            "action_threshold_rule": (
                "development_oof"
            ),
            "warning": (
                "not a calibrated probability"
            ),
        },
        "monitoring_reference": {
            "rows": 40991,
            "prevalence": 0.1008,
        },
    }

    return ModelService(
        model=FakeBinaryModel(),
        model_metadata={
            "best_config_id": "xgboost_03",
            "model_family": "xgboost",
        },
        deployment_config=config,
        telemetry=PredictionTelemetry(),
    )


def _features():
    return {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_weekday": 2,
        "purchase_hour": 10,
        "purchase_is_weekend": 0,
        "purchase_month_sin": 0.0,
        "purchase_month_cos": -1.0,
        "purchase_weekday_sin": 0.5,
        "purchase_weekday_cos": -0.5,
        "purchase_hour_sin": 0.5,
        "purchase_hour_cos": -0.5,
        "approval_lag_hours": 2.0,
        "promised_delivery_days": 10.0,
        "item_count": 1,
        "unique_products": 1,
        "seller_count": 1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "order_total_value": 120.0,
        "mean_item_price": 100.0,
        "max_item_price": 100.0,
        "freight_ratio": 0.1667,
        "payment_records": 1,
        "payment_value": 120.0,
        "max_installments": 1,
        "payment_type_count": 1,
        "mean_product_weight_g": 500.0,
        "total_product_weight_g": 500.0,
        "mean_product_volume_cm3": 1000.0,
        "total_product_volume_cm3": 1000.0,
        "category_count": 1,
        "mean_product_photos": 2.0,
        "seller_state_count": 1,
        "mean_distance_km": 300.0,
        "max_distance_km": 300.0,
        "min_distance_km": 300.0,
        "all_sellers_same_state": 0,
        "same_state_seller_share": 0.0,
        "customer_state": "sp",
        "primary_payment_type": "credit_card",
        "dominant_product_category": "health_beauty",
    }


def test_health_model_info_and_predict():
    app = create_app(
        service=_service(),
        settings=Settings(
            max_batch_size=10,
        ),
    )

    with TestClient(app) as client:
        health = client.get(
            "/health"
        )
        assert health.status_code == 200
        assert health.json()[
            "model_loaded"
        ] is True

        info = client.get(
            "/model-info"
        )
        assert info.status_code == 200
        assert info.json()[
            "calibration_status"
        ] == "not_calibrated"

        response = client.post(
            "/predict",
            json={
                "order_id": "order-123",
                "features": _features(),
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload[
            "order_id"
        ] == "order-123"
        assert payload[
            "calibrated_probability"
        ] is False
        assert payload[
            "risk_band"
        ] in {
            "low",
            "medium",
            "high",
            "critical",
        }

        monitoring = client.get(
            "/monitoring/snapshot"
        )

        assert monitoring.status_code == 200
        assert (
            monitoring.json()[
                "runtime"
            ]["total_predictions"]
            == 1
        )


def test_invalid_feature_contract_returns_422():
    app = create_app(
        service=_service(),
        settings=Settings(),
    )

    payload = _features()
    payload[
        "promised_delivery_days"
    ] = -1

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": payload,
            },
        )

    assert response.status_code == 422


def test_batch_prediction():
    app = create_app(
        service=_service(),
        settings=Settings(
            max_batch_size=10,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {
                        "order_id": "a",
                        "features": _features(),
                    },
                    {
                        "order_id": "b",
                        "features": _features(),
                    },
                ]
            },
        )

    assert response.status_code == 200
    assert len(
        response.json()[
            "predictions"
        ]
    ) == 2
