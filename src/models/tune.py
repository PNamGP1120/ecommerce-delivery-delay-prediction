from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    find_project_root,
)
from src.features.preprocessing import (
    select_model_matrix,
)
from src.models.evaluate import (
    classification_metrics,
    find_best_f1_threshold,
    predict_positive_probability,
)
from src.models.temporal_cv import (
    fold_summary,
    make_expanding_window_folds,
)
from src.models.thresholds import (
    threshold_for_minimum_recall,
)
from src.models.tuning_registry import (
    TuningConfig,
    build_pipeline,
    get_tuning_configs,
)


def load_development_data(
    project_root: Path,
) -> pd.DataFrame:
    processed = project_root / "data" / "processed"

    train = pd.read_parquet(
        processed / "train.parquet"
    )
    validation = pd.read_parquet(
        processed / "validation.parquet"
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

    assert len(development) == len(train) + len(validation)
    assert development["order_id"].is_unique

    return development


def cross_validate_config(
    development: pd.DataFrame,
    folds,
    config: TuningConfig,
    *,
    collect_oof: bool = False,
):
    ordered = (
        development.sort_values(
            ["prediction_timestamp", "order_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    fold_rows = []
    oof_rows = []

    for fold in folds:
        train_frame = ordered.iloc[
            fold.train_index
        ]
        validation_frame = ordered.iloc[
            fold.validation_index
        ]

        X_train = select_model_matrix(
            train_frame
        )
        y_train = train_frame[
            TARGET_COLUMN
        ].copy()

        X_validation = select_model_matrix(
            validation_frame
        )
        y_validation = validation_frame[
            TARGET_COLUMN
        ].copy()

        pipeline = build_pipeline(
            config,
            y_train=y_train,
        )

        started = time.perf_counter()
        pipeline.fit(
            X_train,
            y_train,
        )
        fit_seconds = (
            time.perf_counter() - started
        )

        probability = predict_positive_probability(
            pipeline,
            X_validation,
        )

        metrics = classification_metrics(
            y_validation,
            probability,
            threshold=0.50,
        )

        fold_rows.append(
            {
                "config_id": config.config_id,
                "model_family": config.model_family,
                "fold": fold.fold,
                "fit_seconds": fit_seconds,
                "train_rows": len(train_frame),
                "validation_rows": len(
                    validation_frame
                ),
                "train_late_rate": y_train.mean(),
                "validation_late_rate": (
                    y_validation.mean()
                ),
                **metrics,
            }
        )

        if collect_oof:
            oof_rows.append(
                pd.DataFrame(
                    {
                        "order_id": validation_frame[
                            "order_id"
                        ].to_numpy(),
                        "prediction_timestamp": (
                            validation_frame[
                                "prediction_timestamp"
                            ].to_numpy()
                        ),
                        "late_delivery": (
                            y_validation.to_numpy()
                        ),
                        "probability": probability,
                        "fold": fold.fold,
                    }
                )
            )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    oof = (
        pd.concat(
            oof_rows,
            ignore_index=True,
        )
        if oof_rows
        else None
    )

    return fold_metrics, oof


def summarize_search(
    fold_metrics: pd.DataFrame,
    configs: list[TuningConfig],
) -> pd.DataFrame:
    params_lookup = {
        config.config_id: json.dumps(
            config.params,
            sort_keys=True,
        )
        for config in configs
    }

    summary = (
        fold_metrics.groupby(
            ["config_id", "model_family"],
            as_index=False,
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            min_pr_auc=("pr_auc", "min"),
            max_pr_auc=("pr_auc", "max"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_recall=("recall", "mean"),
            mean_precision=("precision", "mean"),
            mean_f1=("f1", "mean"),
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    summary["std_pr_auc"] = (
        summary["std_pr_auc"].fillna(0.0)
    )

    summary["params"] = summary[
        "config_id"
    ].map(params_lookup)

    summary = (
        summary.sort_values(
            [
                "mean_pr_auc",
                "std_pr_auc",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return summary


def run_search(
    development: pd.DataFrame,
    *,
    quick: bool = False,
):
    folds = make_expanding_window_folds(
        development,
        n_splits=4,
        initial_train_fraction=0.50,
    )

    configs = get_tuning_configs(
        quick=quick
    )

    all_fold_metrics = []

    print(
        f"Temporal CV: {len(folds)} folds"
    )
    print(
        f"Search configs: {len(configs)}"
    )

    for index, config in enumerate(
        configs,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(configs):02d}] "
            f"{config.config_id}"
        )

        fold_metrics, _ = (
            cross_validate_config(
                development,
                folds,
                config,
            )
        )

        all_fold_metrics.append(
            fold_metrics
        )

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    search_summary = summarize_search(
        fold_metrics,
        configs,
    )

    return (
        folds,
        configs,
        fold_metrics,
        search_summary,
    )


def refit_best_config(
    development: pd.DataFrame,
    best_config: TuningConfig,
):
    X_development = select_model_matrix(
        development
    )
    y_development = development[
        TARGET_COLUMN
    ].copy()

    pipeline = build_pipeline(
        best_config,
        y_train=y_development,
    )

    pipeline.fit(
        X_development,
        y_development,
    )

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run one representative configuration "
            "per model family."
        ),
    )

    args = parser.parse_args()

    project_root = find_project_root()

    metrics_dir = (
        project_root / "reports" / "metrics"
    )
    models_dir = (
        project_root / "models"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    development = load_development_data(
        project_root
    )

    (
        folds,
        configs,
        fold_metrics,
        search_summary,
    ) = run_search(
        development,
        quick=args.quick,
    )

    print()
    print("Top temporal-CV configurations:")
    print(
        search_summary.head(12)[
            [
                "config_id",
                "model_family",
                "mean_pr_auc",
                "std_pr_auc",
                "min_pr_auc",
                "mean_roc_auc",
                "mean_recall",
                "mean_precision",
                "total_fit_seconds",
            ]
        ].to_string(index=False)
    )

    best_config_id = str(
        search_summary.iloc[0][
            "config_id"
        ]
    )

    config_lookup = {
        config.config_id: config
        for config in configs
    }

    best_config = config_lookup[
        best_config_id
    ]

    best_fold_metrics, best_oof = (
        cross_validate_config(
            development,
            folds,
            best_config,
            collect_oof=True,
        )
    )

    best_f1_threshold = (
        find_best_f1_threshold(
            best_oof["late_delivery"],
            best_oof["probability"],
        )
    )

    recall_50_threshold = (
        threshold_for_minimum_recall(
            best_oof["late_delivery"],
            best_oof["probability"],
            minimum_recall=0.50,
        )
    )

    best_pipeline = refit_best_config(
        development,
        best_config,
    )

    fold_summary_frame = fold_summary(
        development,
        folds,
        target_column=TARGET_COLUMN,
    )

    fold_summary_frame.to_csv(
        metrics_dir
        / "temporal_cv_folds.csv",
        index=False,
    )

    fold_metrics.to_csv(
        metrics_dir
        / "temporal_tuning_fold_metrics.csv",
        index=False,
    )

    search_summary.to_csv(
        metrics_dir
        / "temporal_tuning_summary.csv",
        index=False,
    )

    best_oof.to_parquet(
        metrics_dir
        / "best_tuned_oof_predictions.parquet",
        index=False,
    )

    threshold_frame = pd.DataFrame(
        [
            {
                "threshold_rule": "best_oof_f1",
                "threshold": (
                    best_f1_threshold.threshold
                ),
                "precision": (
                    best_f1_threshold.precision
                ),
                "recall": (
                    best_f1_threshold.recall
                ),
                "f1": best_f1_threshold.f1,
            },
            {
                "threshold_rule": (
                    recall_50_threshold.rule
                ),
                "threshold": (
                    recall_50_threshold.threshold
                ),
                "precision": (
                    recall_50_threshold.precision
                ),
                "recall": (
                    recall_50_threshold.recall
                ),
                "f1": recall_50_threshold.f1,
            },
        ]
    )

    threshold_frame.to_csv(
        metrics_dir
        / "tuned_operating_thresholds.csv",
        index=False,
    )

    joblib.dump(
        best_pipeline,
        models_dir
        / "best_tuned_candidate.joblib",
    )

    metadata = {
        "status": (
            "development_candidate_not_final_holdout"
        ),
        "best_config_id": (
            best_config.config_id
        ),
        "model_family": (
            best_config.model_family
        ),
        "params": best_config.params,
        "selection_rule": (
            "highest mean temporal-CV PR-AUC; "
            "lower PR-AUC std as tie-break"
        ),
        "mean_cv_pr_auc": float(
            search_summary.iloc[0][
                "mean_pr_auc"
            ]
        ),
        "std_cv_pr_auc": float(
            search_summary.iloc[0][
                "std_pr_auc"
            ]
        ),
        "min_cv_pr_auc": float(
            search_summary.iloc[0][
                "min_pr_auc"
            ]
        ),
        "feature_count": len(
            MODEL_FEATURE_COLUMNS
        ),
        "development_rows": len(
            development
        ),
        "development_end": str(
            development[
                "prediction_timestamp"
            ].max()
        ),
        "best_oof_f1_threshold": float(
            best_f1_threshold.threshold
        ),
        "recall_50_threshold": float(
            recall_50_threshold.threshold
        ),
        "note": (
            "The original test split was already opened "
            "during Phase 4A. Phase 4B therefore does not "
            "use it for tuning or default evaluation."
        ),
    }

    with (
        models_dir
        / "best_tuned_candidate_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        "Best configuration:",
        best_config.config_id,
    )
    print(
        "Model family:",
        best_config.model_family,
    )
    print(
        "Mean CV PR-AUC:",
        f"{metadata['mean_cv_pr_auc']:.6f}",
    )
    print(
        "Std CV PR-AUC:",
        f"{metadata['std_cv_pr_auc']:.6f}",
    )
    print()
    print("OOF operating thresholds:")
    print(
        threshold_frame.to_string(
            index=False
        )
    )
    print()
    print(
        "✓ Tuned development candidate saved."
    )
    print(
        "✓ Original test split was NOT used "
        "by Phase 4B."
    )


if __name__ == "__main__":
    main()
