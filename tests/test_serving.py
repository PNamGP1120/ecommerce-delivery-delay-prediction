from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.telemetry import (
    PredictionTelemetry,
)
from src.serving.deployment import (
    build_deployment_config_data,
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


def _deployment_config():
    return {
        "model": {
            "candidate": "xgboost_03",
            "family": "xgboost",
            "sha256": "a" * 64,
            "feature_count": 41,
        },
        "risk_score": {
            "semantics": "ranking score",
            "calibrated_probability": False,
            "risk_band_source": "OOF quantiles",
            "bands": {
                "low_max": 0.20,
                "medium_max": 0.40,
                "high_max": 0.60,
            },
            "action_threshold": 0.50,
            "action_threshold_rule": "test_rule",
            "warning": "not calibrated",
        },
        "monitoring_reference": {
            "rows": 100,
            "prevalence": 0.10,
        },
    }


def _feature_row():
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
        "customer_state": "SP",
        "primary_payment_type": "credit_card",
        "dominant_product_category": "health_beauty",
    }


def test_model_service_returns_ranking_output():
    telemetry = PredictionTelemetry()

    service = ModelService(
        model=FakeBinaryModel(),
        model_metadata={
            "best_config_id": "xgboost_03",
            "model_family": "xgboost",
        },
        deployment_config=(
            _deployment_config()
        ),
        telemetry=telemetry,
    )

    result = service.predict(
        [_feature_row()],
        request_id="test-request",
    )[0]

    assert 0 <= result.risk_score <= 1
    assert result.risk_band in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert result.requires_review is True

    snapshot = telemetry.snapshot()

    assert snapshot[
        "total_predictions"
    ] == 1
    assert snapshot[
        "total_requests"
    ] == 1


def test_deployment_config_uses_oof_quantiles(
    tmp_path: Path,
):
    model_path = (
        tmp_path / "model.joblib"
    )
    model_path.write_bytes(
        b"fake-model"
    )

    oof = pd.DataFrame(
        {
            "late_delivery": (
                [0, 0, 1, 0, 1]
                * 20
            ),
            "probability": np.linspace(
                0.05,
                0.90,
                100,
            ),
        }
    )

    thresholds = pd.DataFrame(
        [
            {
                "threshold_rule": (
                    "best_oof_f1"
                ),
                "threshold": 0.55,
            },
            {
                "threshold_rule": (
                    "maximize_precision_subject_to_"
                    "recall>=0.50"
                ),
                "threshold": 0.50,
            },
        ]
    )

    config = build_deployment_config_data(
        model_path=model_path,
        model_metadata={
            "best_config_id": (
                "xgboost_03"
            ),
            "model_family": "xgboost",
        },
        oof_predictions=oof,
        operating_thresholds=thresholds,
    )

    assert config[
        "risk_score"
    ]["action_threshold"] == 0.50

    bands = config[
        "risk_score"
    ]["bands"]

    assert (
        bands["low_max"]
        < bands["medium_max"]
        < bands["high_max"]
    )
