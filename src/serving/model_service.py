from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import (
    MODEL_FEATURE_COLUMNS,
)
from src.models.evaluate import (
    predict_positive_probability,
)
from src.monitoring.telemetry import (
    PredictionTelemetry,
)


logger = logging.getLogger(
    "delivery_delay.serving"
)


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    risk_band: str
    requires_review: bool


class ModelService:
    def __init__(
        self,
        *,
        model,
        model_metadata: dict,
        deployment_config: dict,
        telemetry: PredictionTelemetry | None = None,
        action_threshold_override: float | None = None,
    ) -> None:
        self.model = model
        self.model_metadata = (
            model_metadata
        )
        self.deployment_config = (
            deployment_config
        )
        self.telemetry = (
            telemetry
            or PredictionTelemetry()
        )

        risk_config = (
            deployment_config[
                "risk_score"
            ]
        )

        self.action_threshold = (
            float(
                action_threshold_override
            )
            if (
                action_threshold_override
                is not None
            )
            else float(
                risk_config[
                    "action_threshold"
                ]
            )
        )

        if not (
            0
            <= self.action_threshold
            <= 1
        ):
            raise ValueError(
                "Action threshold must be "
                "between 0 and 1."
            )

        self.model_version = (
            f"{deployment_config['model']['candidate']}:"
            f"{deployment_config['model']['sha256'][:12]}"
        )

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: Path,
        model_metadata_path: Path,
        deployment_config_path: Path,
        telemetry: PredictionTelemetry | None = None,
        action_threshold_override: float | None = None,
    ) -> "ModelService":
        missing = [
            path
            for path in [
                model_path,
                model_metadata_path,
                deployment_config_path,
            ]
            if not path.exists()
        ]

        if missing:
            formatted = ", ".join(
                str(path)
                for path in missing
            )
            raise FileNotFoundError(
                "Required serving artifact(s) "
                f"not found: {formatted}"
            )

        model = joblib.load(
            model_path
        )

        with model_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        with deployment_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            deployment_config = (
                json.load(file)
            )

        return cls(
            model=model,
            model_metadata=metadata,
            deployment_config=deployment_config,
            telemetry=telemetry,
            action_threshold_override=(
                action_threshold_override
            ),
        )

    @property
    def candidate(self) -> str:
        return str(
            self.deployment_config[
                "model"
            ]["candidate"]
        )

    @property
    def model_family(self) -> str:
        return str(
            self.deployment_config[
                "model"
            ]["family"]
        )

    @property
    def feature_count(self) -> int:
        return int(
            self.deployment_config[
                "model"
            ]["feature_count"]
        )

    def risk_band(
        self,
        score: float,
    ) -> str:
        bands = self.deployment_config[
            "risk_score"
        ]["bands"]

        if score <= bands["low_max"]:
            return "low"
        if score <= bands["medium_max"]:
            return "medium"
        if score <= bands["high_max"]:
            return "high"

        return "critical"

    def predict(
        self,
        feature_rows: list[dict],
        *,
        request_id: str,
    ) -> list[PredictionResult]:
        started = time.perf_counter()

        frame = pd.DataFrame(
            feature_rows
        )

        missing_columns = set(
            MODEL_FEATURE_COLUMNS
        ).difference(frame.columns)

        if missing_columns:
            raise ValueError(
                "Missing model feature columns: "
                f"{sorted(missing_columns)}"
            )

        frame = frame[
            MODEL_FEATURE_COLUMNS
        ].copy()

        frame = frame.replace(
            {
                None: np.nan,
            }
        )

        missing_feature_values = int(
            frame.isna().sum().sum()
        )

        try:
            scores = (
                predict_positive_probability(
                    self.model,
                    frame,
                )
            )
        except Exception:
            self.telemetry.record_failure()
            raise

        score_list = [
            float(score)
            for score in scores
        ]
        bands = [
            self.risk_band(score)
            for score in score_list
        ]
        review = [
            score
            >= self.action_threshold
            for score in score_list
        ]

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000

        self.telemetry.record_batch(
            scores=score_list,
            bands=bands,
            requires_review=review,
            latency_ms=latency_ms,
            missing_feature_values=(
                missing_feature_values
            ),
        )

        logger.info(
            "prediction_completed",
            extra={
                "event": (
                    "prediction_completed"
                ),
                "request_id": request_id,
                "prediction_count": len(
                    score_list
                ),
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
                "mean_score": (
                    float(
                        np.mean(
                            score_list
                        )
                    )
                    if score_list
                    else None
                ),
                "review_count": sum(
                    review
                ),
            },
        )

        return [
            PredictionResult(
                risk_score=score,
                risk_band=band,
                requires_review=flag,
            )
            for score, band, flag in zip(
                score_list,
                bands,
                review,
                strict=True,
            )
        ]
