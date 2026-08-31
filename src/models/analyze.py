from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    find_project_root,
)
from src.features.preprocessing import (
    select_model_matrix,
)
from src.models.analysis_utils import (
    add_analysis_segments,
    calibration_table,
    ranking_metrics,
    segment_metrics,
    threshold_metrics,
    threshold_sweep,
)
from src.models.drift import drift_summary
from src.models.evaluate import (
    predict_positive_probability,
)
from src.models.explain import (
    native_xgboost_importance,
    raw_permutation_importance,
    shap_global_importance,
)


def _load_thresholds(
    metrics_dir: Path,
) -> dict[str, float]:
    thresholds = pd.read_csv(
        metrics_dir
        / "tuned_operating_thresholds.csv"
    )

    return {
        str(row["threshold_rule"]): float(
            row["threshold"]
        )
        for _, row in thresholds.iterrows()
    }


def _save_threshold_plot(
    sweep: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        sweep["threshold"],
        sweep["precision"],
        marker="o",
        label="Precision",
    )
    ax.plot(
        sweep["threshold"],
        sweep["recall"],
        marker="o",
        label="Recall",
    )
    ax.plot(
        sweep["threshold"],
        sweep["f1"],
        marker="o",
        label="F1",
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Observed Holdout Threshold Trade-off"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_calibration_plot(
    calibration: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    ax.plot(
        calibration[
            "mean_predicted_probability"
        ],
        calibration[
            "observed_late_rate"
        ],
        marker="o",
        label="XGBoost",
    )

    ax.set_xlabel(
        "Mean predicted probability"
    )
    ax.set_ylabel(
        "Observed late-delivery rate"
    )
    ax.set_title(
        "Observed Holdout Calibration"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_top_importance_plot(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    path: Path,
    top_n: int = 20,
) -> None:
    top = (
        frame.head(top_n)
        .sort_values(
            value_column,
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        top["feature"],
        top[value_column],
    )

    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    project_root = find_project_root()

    processed_dir = (
        project_root / "data" / "processed"
    )
    metrics_dir = (
        project_root / "reports" / "metrics"
    )
    figures_dir = (
        project_root / "reports" / "figures"
    )
    models_dir = (
        project_root / "models"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / "best_tuned_candidate.joblib"
    )
    metadata_path = (
        models_dir
        / "best_tuned_candidate_metadata.json"
    )

    model = joblib.load(
        model_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    train = pd.read_parquet(
        processed_dir / "train.parquet"
    )
    validation = pd.read_parquet(
        processed_dir / "validation.parquet"
    )
    test = pd.read_parquet(
        processed_dir / "test.parquet"
    )

    development = (
        pd.concat(
            [train, validation],
            ignore_index=True,
        )
        .sort_values(
            ["prediction_timestamp", "order_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    oof = pd.read_parquet(
        metrics_dir
        / "best_tuned_oof_predictions.parquet"
    )

    thresholds = _load_thresholds(
        metrics_dir
    )

    X_test = select_model_matrix(
        test
    )
    y_test = test[
        TARGET_COLUMN
    ].copy()

    test_probability = (
        predict_positive_probability(
            model,
            X_test,
        )
    )

    test_scored = (
        test.copy()
    )
    test_scored["probability"] = (
        test_probability
    )
    test_scored = (
        add_analysis_segments(
            test_scored
        )
    )

    # 1. Ranking diagnostics.
    evaluation_rows = [
        {
            "dataset": (
                "development_oof"
            ),
            "evaluation_status": (
                "temporal_oof"
            ),
            **ranking_metrics(
                oof["late_delivery"],
                oof["probability"],
            ),
        },
        {
            "dataset": (
                "phase4a_test"
            ),
            "evaluation_status": (
                "observed_holdout_diagnostic"
            ),
            **ranking_metrics(
                y_test,
                test_probability,
            ),
        },
    ]

    ranking_frame = pd.DataFrame(
        evaluation_rows
    )
    ranking_frame.to_csv(
        metrics_dir
        / "phase5_ranking_metrics.csv",
        index=False,
    )

    # 2. Threshold diagnostics.
    threshold_rows = []

    named_thresholds = {
        "default_0.50": 0.50,
        **thresholds,
    }

    for name, threshold in (
        named_thresholds.items()
    ):
        threshold_rows.append(
            {
                "threshold_rule": name,
                **threshold_metrics(
                    y_test,
                    test_probability,
                    threshold,
                ),
            }
        )

    threshold_eval = pd.DataFrame(
        threshold_rows
    )
    threshold_eval.to_csv(
        metrics_dir
        / "phase5_test_threshold_metrics.csv",
        index=False,
    )

    sweep = threshold_sweep(
        y_test,
        test_probability,
    )
    sweep.to_csv(
        metrics_dir
        / "phase5_threshold_sweep.csv",
        index=False,
    )
    _save_threshold_plot(
        sweep,
        figures_dir
        / "15_threshold_tradeoff.png",
    )

    # 3. Calibration.
    calibration = calibration_table(
        y_test,
        test_probability,
        n_bins=10,
    )
    calibration.to_csv(
        metrics_dir
        / "phase5_calibration.csv",
        index=False,
    )
    _save_calibration_plot(
        calibration,
        figures_dir
        / "16_calibration_curve.png",
    )

    # 4. Fold-level robustness for selected config.
    fold_metrics = pd.read_csv(
        metrics_dir
        / "temporal_tuning_fold_metrics.csv"
    )
    best_config_id = metadata[
        "best_config_id"
    ]

    best_folds = (
        fold_metrics[
            fold_metrics["config_id"]
            == best_config_id
        ]
        .copy()
        .sort_values("fold")
    )

    best_folds["pr_auc_lift"] = (
        best_folds["pr_auc"]
        / best_folds[
            "validation_late_rate"
        ]
    )

    best_folds.to_csv(
        metrics_dir
        / "phase5_temporal_robustness.csv",
        index=False,
    )

    # 5. Segment diagnostics.
    segment_specs = [
        (
            "customer_state",
            "phase5_segment_customer_state.csv",
            100,
            10,
        ),
        (
            "distance_band",
            "phase5_segment_distance.csv",
            100,
            10,
        ),
        (
            "promise_band",
            "phase5_segment_promise_window.csv",
            100,
            10,
        ),
        (
            "approval_lag_band",
            "phase5_segment_approval_lag.csv",
            100,
            10,
        ),
        (
            "prediction_month",
            "phase5_segment_month.csv",
            100,
            10,
        ),
    ]

    for (
        segment_column,
        filename,
        minimum_rows,
        minimum_positives,
    ) in segment_specs:
        segment_frame = segment_metrics(
            test_scored,
            segment_column=segment_column,
            target_column=TARGET_COLUMN,
            probability_column="probability",
            minimum_rows=minimum_rows,
            minimum_positives=minimum_positives,
        )
        segment_frame.to_csv(
            metrics_dir / filename,
            index=False,
        )

    # 6. Drift diagnostics.
    drift = drift_summary(
        development,
        test,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    drift.to_csv(
        metrics_dir
        / "phase5_feature_drift.csv",
        index=False,
    )

    # 7. Feature importance.
    native_importance = (
        native_xgboost_importance(
            model
        )
    )
    native_importance.to_csv(
        metrics_dir
        / "phase5_native_feature_importance.csv",
        index=False,
    )
    _save_top_importance_plot(
        native_importance,
        value_column="importance",
        title=(
            "XGBoost Native Feature Importance"
        ),
        path=figures_dir
        / "17_native_feature_importance.png",
    )

    permutation = (
        raw_permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=(
                args.permutation_repeats
            ),
        )
    )
    permutation.to_csv(
        metrics_dir
        / "phase5_permutation_importance.csv",
        index=False,
    )
    _save_top_importance_plot(
        permutation,
        value_column="importance_mean",
        title=(
            "Observed Holdout Permutation Importance"
        ),
        path=figures_dir
        / "18_permutation_importance.png",
    )

    shap_status = "skipped"

    if not args.skip_shap:
        try:
            (
                shap_importance,
                shap_values,
                feature_names,
            ) = shap_global_importance(
                model,
                X_test,
                sample_size=2_000,
            )

            shap_importance.to_csv(
                metrics_dir
                / "phase5_shap_importance.csv",
                index=False,
            )

            _save_top_importance_plot(
                shap_importance,
                value_column="mean_abs_shap",
                title=(
                    "Mean Absolute SHAP Importance"
                ),
                path=figures_dir
                / "19_shap_importance.png",
            )

            shap_status = "complete"
        except ImportError as exc:
            print(exc)
            shap_status = (
                "not_installed"
            )

    # 8. Save scored observed holdout for Phase 5 analysis.
    test_scored[
        [
            "order_id",
            "prediction_timestamp",
            TARGET_COLUMN,
            "probability",
            "customer_state",
            "mean_distance_km",
            "promised_delivery_days",
            "approval_lag_hours",
            "distance_band",
            "promise_band",
            "approval_lag_band",
            "prediction_month",
        ]
    ].to_parquet(
        metrics_dir
        / "phase5_test_predictions.parquet",
        index=False,
    )

    phase5_metadata = {
        "candidate": metadata[
            "best_config_id"
        ],
        "model_family": metadata[
            "model_family"
        ],
        "development_rows": len(
            development
        ),
        "test_rows": len(test),
        "test_status": (
            "observed_holdout_diagnostic_not_pristine"
        ),
        "selection_policy": (
            "No model or hyperparameter changes are made "
            "from Phase 5 test results."
        ),
        "shap_status": shap_status,
    }

    with (
        metrics_dir
        / "phase5_evaluation_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            phase5_metadata,
            file,
            indent=2,
        )

    print()
    print("Phase 5 ranking metrics:")
    print(
        ranking_frame.to_string(
            index=False
        )
    )

    print()
    print(
        "Observed holdout threshold diagnostics:"
    )
    print(
        threshold_eval.to_string(
            index=False
        )
    )

    print()
    print(
        "Top feature drift:"
    )
    print(
        drift.head(12)[
            [
                "feature",
                "feature_type",
                "psi",
                "drift_flag",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Top permutation importance:"
    )
    print(
        permutation.head(15).to_string(
            index=False
        )
    )

    print()
    print(
        "✓ Phase 5 analysis artifacts saved."
    )
    print(
        "✓ No Phase 5 test metric was used "
        "for model selection or tuning."
    )


if __name__ == "__main__":
    main()
