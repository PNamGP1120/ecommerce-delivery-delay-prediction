from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (
    MODEL_FEATURE_COLUMNS,
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_deployment_config_data(
    *,
    model_path: Path,
    model_metadata: dict,
    oof_predictions: pd.DataFrame,
    operating_thresholds: pd.DataFrame,
) -> dict:
    scores = pd.to_numeric(
        oof_predictions["probability"],
        errors="coerce",
    ).dropna()

    if scores.empty:
        raise ValueError(
            "OOF predictions contain no valid scores."
        )

    quantiles = {
        "q05": float(
            scores.quantile(0.05)
        ),
        "q25": float(
            scores.quantile(0.25)
        ),
        "q50": float(
            scores.quantile(0.50)
        ),
        "q75": float(
            scores.quantile(0.75)
        ),
        "q80": float(
            scores.quantile(0.80)
        ),
        "q90": float(
            scores.quantile(0.90)
        ),
        "q95": float(
            scores.quantile(0.95)
        ),
        "q99": float(
            scores.quantile(0.99)
        ),
    }

    preferred_rule = (
        "maximize_precision_subject_to_recall>=0.50"
    )

    selected = operating_thresholds[
        operating_thresholds[
            "threshold_rule"
        ].astype(str)
        == preferred_rule
    ]

    if selected.empty:
        selected = operating_thresholds[
            operating_thresholds[
                "threshold_rule"
            ].astype(str)
            == "best_oof_f1"
        ]

    if selected.empty:
        raise ValueError(
            "No supported operating threshold "
            "was found."
        )

    threshold_row = selected.iloc[0]

    return {
        "model": {
            "candidate": model_metadata[
                "best_config_id"
            ],
            "family": model_metadata[
                "model_family"
            ],
            "sha256": sha256_file(
                model_path
            ),
            "feature_count": len(
                MODEL_FEATURE_COLUMNS
            ),
        },
        "risk_score": {
            "semantics": (
                "Positive-class model score used for "
                "ranking risk. It is not a calibrated "
                "late-delivery probability."
            ),
            "calibrated_probability": False,
            "risk_band_source": (
                "development temporal OOF score quantiles"
            ),
            "bands": {
                "low_max": quantiles[
                    "q50"
                ],
                "medium_max": quantiles[
                    "q80"
                ],
                "high_max": quantiles[
                    "q95"
                ],
            },
            "action_threshold": float(
                threshold_row[
                    "threshold"
                ]
            ),
            "action_threshold_rule": str(
                threshold_row[
                    "threshold_rule"
                ]
            ),
            "warning": (
                "Phase 5 showed temporal threshold "
                "instability and poor calibration. "
                "Treat requires_review as a configurable "
                "operational policy, not a probability "
                "guarantee."
            ),
        },
        "monitoring_reference": {
            "source": (
                "best_tuned_oof_predictions.parquet"
            ),
            "rows": int(
                len(oof_predictions)
            ),
            "prevalence": float(
                oof_predictions[
                    "late_delivery"
                ].mean()
            ),
            "score_mean": float(
                scores.mean()
            ),
            "score_std": float(
                scores.std()
            ),
            "score_quantiles": quantiles,
        },
    }


def write_deployment_config(
    config: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )
