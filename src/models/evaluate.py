from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLD = 0.50


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def predict_positive_probability(model, X) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model must implement predict_proba for PR-AUC evaluation."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Expected binary predict_proba output with two columns."
        )

    return probabilities[:, 1]


def classification_metrics(
    y_true,
    y_probability,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability, dtype="float64")

    y_pred = (y_probability >= threshold).astype("int8")

    return {
        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "positive_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }


def confusion_matrix_frame(
    y_true,
    y_probability,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    y_pred = (
        np.asarray(y_probability) >= threshold
    ).astype("int8")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return pd.DataFrame(
        matrix,
        index=["actual_on_time", "actual_late"],
        columns=["pred_on_time", "pred_late"],
    )


def find_best_f1_threshold(
    y_true,
    y_probability,
) -> ThresholdResult:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if thresholds.size == 0:
        return ThresholdResult(
            threshold=DEFAULT_THRESHOLD,
            precision=float(precision[0]),
            recall=float(recall[0]),
            f1=0.0,
        )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.nanargmax(f1))

    return ThresholdResult(
        threshold=float(thresholds[best_index]),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        f1=float(f1[best_index]),
    )


def threshold_table(
    y_true,
    y_probability,
    thresholds: Iterable[float] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ),
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        metrics = classification_metrics(
            y_true,
            y_probability,
            threshold=threshold,
        )
        rows.append(metrics)

    return pd.DataFrame(rows)
