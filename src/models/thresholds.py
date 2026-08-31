from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class OperatingThreshold:
    threshold: float
    precision: float
    recall: float
    f1: float
    rule: str


def threshold_for_minimum_recall(
    y_true,
    y_probability,
    *,
    minimum_recall: float = 0.50,
) -> OperatingThreshold:
    if not (0 < minimum_recall <= 1):
        raise ValueError(
            "minimum_recall must be in (0, 1]."
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if thresholds.size == 0:
        raise ValueError(
            "No classification threshold could be computed."
        )

    precision = precision[:-1]
    recall = recall[:-1]

    feasible = np.where(
        recall >= minimum_recall
    )[0]

    if feasible.size == 0:
        best_index = int(np.argmax(recall))
    else:
        feasible_precision = precision[feasible]
        best_index = int(
            feasible[
                np.argmax(feasible_precision)
            ]
        )

    denominator = (
        precision[best_index]
        + recall[best_index]
    )
    f1 = (
        2
        * precision[best_index]
        * recall[best_index]
        / denominator
        if denominator > 0
        else 0.0
    )

    return OperatingThreshold(
        threshold=float(thresholds[best_index]),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        f1=float(f1),
        rule=(
            f"maximize_precision_subject_to_"
            f"recall>={minimum_recall:.2f}"
        ),
    )
