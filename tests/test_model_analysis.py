import numpy as np
import pandas as pd

from src.models.analysis_utils import (
    add_analysis_segments,
    calibration_table,
    ranking_metrics,
    segment_metrics,
    threshold_metrics,
)
from src.models.drift import (
    categorical_psi,
    numeric_psi,
)


def test_ranking_metrics_include_pr_auc_lift():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    result = ranking_metrics(y, p)

    assert result["pr_auc"] == 1.0
    assert result["prevalence"] == 0.5
    assert result["pr_auc_lift"] == 2.0


def test_threshold_metrics_confusion_counts_sum():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.7, 0.8, 0.2])

    result = threshold_metrics(
        y,
        p,
        0.5,
    )

    assert (
        result["tp"]
        + result["fp"]
        + result["tn"]
        + result["fn"]
        == 4
    )


def test_numeric_psi_zero_for_identical_distribution():
    reference = pd.Series(
        np.arange(100, dtype=float)
    )

    psi = numeric_psi(
        reference,
        reference.copy(),
    )

    assert abs(psi) < 1e-12


def test_categorical_psi_detects_shift():
    reference = pd.Series(
        ["A"] * 90 + ["B"] * 10
    )
    comparison = pd.Series(
        ["A"] * 10 + ["B"] * 90
    )

    psi = categorical_psi(
        reference,
        comparison,
    )

    assert psi > 0.25


def test_segment_metrics_and_analysis_bands():
    n = 200
    frame = pd.DataFrame(
        {
            "prediction_timestamp": (
                pd.date_range(
                    "2024-01-01",
                    periods=n,
                    freq="h",
                )
            ),
            "customer_state": (
                ["SP"] * 100
                + ["RJ"] * 100
            ),
            "mean_distance_km": (
                np.linspace(
                    10,
                    2500,
                    n,
                )
            ),
            "promised_delivery_days": (
                np.linspace(
                    5,
                    45,
                    n,
                )
            ),
            "approval_lag_hours": (
                np.linspace(
                    0,
                    100,
                    n,
                )
            ),
            "late_delivery": (
                np.arange(n) % 5 == 0
            ).astype("int8"),
            "probability": np.linspace(
                0.01,
                0.99,
                n,
            ),
        }
    )

    segmented = add_analysis_segments(
        frame
    )

    result = segment_metrics(
        segmented,
        segment_column="customer_state",
        target_column="late_delivery",
        probability_column="probability",
        minimum_rows=50,
        minimum_positives=5,
    )

    assert len(result) == 2
    assert {
        "distance_band",
        "promise_band",
        "approval_lag_band",
        "prediction_month",
    }.issubset(segmented.columns)


def test_calibration_table_has_expected_columns():
    y = np.array(
        [0, 0, 0, 1, 1, 1]
    )
    p = np.array(
        [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
    )

    table = calibration_table(
        y,
        p,
        n_bins=3,
    )

    assert {
        "mean_predicted_probability",
        "observed_late_rate",
        "calibration_gap",
    } == set(table.columns)
