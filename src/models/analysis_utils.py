from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPSILON = 1e-6


def ranking_metrics(
    y_true,
    probability,
) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype="float64")

    prevalence = float(y.mean())
    pr_auc = float(
        average_precision_score(y, p)
    )
    roc_auc = float(
        roc_auc_score(y, p)
    )

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift": (
            pr_auc / prevalence
            if prevalence > 0
            else np.nan
        ),
        "roc_auc": roc_auc,
        "brier_score": float(
            brier_score_loss(y, p)
        ),
        "log_loss": float(
            log_loss(
                y,
                np.column_stack([1 - p, p]),
                labels=[0, 1],
            )
        ),
    }


def threshold_metrics(
    y_true,
    probability,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype="float64")
    pred = (p >= threshold).astype("int8")

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "positive_rate": float(
            pred.mean()
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_sweep(
    y_true,
    probability,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(
            0.05,
            0.95,
            19,
        )

    return pd.DataFrame(
        [
            threshold_metrics(
                y_true,
                probability,
                float(threshold),
            )
            for threshold in thresholds
        ]
    )


def calibration_table(
    y_true,
    probability,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype="float64")

    fraction_positive, mean_predicted = (
        calibration_curve(
            y,
            p,
            n_bins=n_bins,
            strategy="quantile",
        )
    )

    return pd.DataFrame(
        {
            "mean_predicted_probability": (
                mean_predicted
            ),
            "observed_late_rate": (
                fraction_positive
            ),
            "calibration_gap": (
                fraction_positive
                - mean_predicted
            ),
        }
    )


def segment_metrics(
    frame: pd.DataFrame,
    *,
    segment_column: str,
    target_column: str,
    probability_column: str,
    minimum_rows: int = 100,
    minimum_positives: int = 5,
) -> pd.DataFrame:
    rows = []

    for segment, group in frame.groupby(
        segment_column,
        dropna=False,
        observed=False,
    ):
        y = group[target_column].to_numpy()
        p = group[
            probability_column
        ].to_numpy()

        positives = int(y.sum())

        if (
            len(group) < minimum_rows
            or positives < minimum_positives
            or positives == len(group)
        ):
            continue

        metrics = ranking_metrics(
            y,
            p,
        )

        rows.append(
            {
                "segment": segment,
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "segment",
                "rows",
                "positives",
                "prevalence",
                "pr_auc",
                "pr_auc_lift",
                "roc_auc",
                "brier_score",
                "log_loss",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["rows", "segment"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def add_analysis_segments(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["prediction_month"] = (
        pd.to_datetime(
            result["prediction_timestamp"]
        )
        .dt.to_period("M")
        .astype(str)
    )

    result["distance_band"] = pd.cut(
        result["mean_distance_km"],
        bins=[
            -np.inf,
            250,
            500,
            1_000,
            2_000,
            np.inf,
        ],
        labels=[
            "<=250 km",
            "250-500 km",
            "500-1000 km",
            "1000-2000 km",
            ">2000 km",
        ],
    )

    result["promise_band"] = pd.cut(
        result["promised_delivery_days"],
        bins=[
            -np.inf,
            10,
            20,
            30,
            40,
            np.inf,
        ],
        labels=[
            "<=10 days",
            "10-20 days",
            "20-30 days",
            "30-40 days",
            ">40 days",
        ],
    )

    result["approval_lag_band"] = pd.cut(
        result["approval_lag_hours"],
        bins=[
            -np.inf,
            1,
            6,
            24,
            72,
            np.inf,
        ],
        labels=[
            "<=1 h",
            "1-6 h",
            "6-24 h",
            "24-72 h",
            ">72 h",
        ],
    )

    return result
