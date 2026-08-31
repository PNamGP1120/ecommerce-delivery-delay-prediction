from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    find_project_root,
)
from src.features.preprocessing import select_model_matrix
from src.models.evaluate import (
    classification_metrics,
    confusion_matrix_frame,
    find_best_f1_threshold,
    predict_positive_probability,
)
from src.models.model_registry import get_model_registry


PRIMARY_METRIC = "pr_auc"


def load_processed_splits(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = project_root / "data" / "processed"

    train = pd.read_parquet(
        processed_dir / "train.parquet"
    )
    validation = pd.read_parquet(
        processed_dir / "validation.parquet"
    )
    test = pd.read_parquet(
        processed_dir / "test.parquet"
    )

    return train, validation, test


def _validate_split(frame: pd.DataFrame, name: str) -> None:
    required = {
        "order_id",
        "prediction_timestamp",
        TARGET_COLUMN,
        *MODEL_FEATURE_COLUMNS,
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )

    assert frame["order_id"].is_unique
    assert frame[TARGET_COLUMN].isin([0, 1]).all()


def train_and_compare_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    _validate_split(train_df, "train")
    _validate_split(validation_df, "validation")

    X_train = select_model_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].copy()

    X_validation = select_model_matrix(validation_df)
    y_validation = validation_df[TARGET_COLUMN].copy()

    registry = get_model_registry()

    fitted_models = {}
    validation_probabilities = {}
    rows = []

    for name, spec in registry.items():
        print(f"Training {name} ...")

        pipeline = spec.pipeline_factory()

        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        probabilities = predict_positive_probability(
            pipeline,
            X_validation,
        )

        metrics = classification_metrics(
            y_validation,
            probabilities,
            threshold=0.50,
        )

        rows.append(
            {
                "model": name,
                "family": spec.family,
                "description": spec.description,
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )

        fitted_models[name] = pipeline
        validation_probabilities[name] = probabilities

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values(
            PRIMARY_METRIC,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        metrics_df,
        fitted_models,
        validation_probabilities,
    )


def select_best_model(
    validation_metrics: pd.DataFrame,
) -> str:
    if validation_metrics.empty:
        raise ValueError(
            "validation_metrics cannot be empty."
        )

    return str(validation_metrics.iloc[0]["model"])


def evaluate_best_model_on_test(
    best_model,
    validation_df: pd.DataFrame,
    validation_probability,
    test_df: pd.DataFrame,
):
    y_validation = validation_df[TARGET_COLUMN].copy()

    threshold_result = find_best_f1_threshold(
        y_validation,
        validation_probability,
    )

    X_test = select_model_matrix(test_df)
    y_test = test_df[TARGET_COLUMN].copy()

    test_probability = predict_positive_probability(
        best_model,
        X_test,
    )

    default_metrics = classification_metrics(
        y_test,
        test_probability,
        threshold=0.50,
    )

    tuned_metrics = classification_metrics(
        y_test,
        test_probability,
        threshold=threshold_result.threshold,
    )

    confusion_default = confusion_matrix_frame(
        y_test,
        test_probability,
        threshold=0.50,
    )

    confusion_tuned = confusion_matrix_frame(
        y_test,
        test_probability,
        threshold=threshold_result.threshold,
    )

    return {
        "threshold_result": threshold_result,
        "test_probability": test_probability,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "confusion_default": confusion_default,
        "confusion_tuned": confusion_tuned,
    }


def main() -> None:
    project_root = find_project_root()

    reports_metrics_dir = (
        project_root / "reports" / "metrics"
    )
    models_dir = project_root / "models"

    reports_metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, validation_df, test_df = (
        load_processed_splits(project_root)
    )

    (
        validation_metrics,
        fitted_models,
        validation_probabilities,
    ) = train_and_compare_models(
        train_df,
        validation_df,
    )

    best_name = select_best_model(
        validation_metrics
    )
    best_model = fitted_models[best_name]

    test_result = evaluate_best_model_on_test(
        best_model=best_model,
        validation_df=validation_df,
        validation_probability=(
            validation_probabilities[best_name]
        ),
        test_df=test_df,
    )

    validation_metrics.to_csv(
        reports_metrics_dir
        / "model_validation_metrics.csv",
        index=False,
    )

    test_metrics = pd.DataFrame(
        [
            {
                "model": best_name,
                "evaluation": "default_threshold",
                **test_result["default_metrics"],
            },
            {
                "model": best_name,
                "evaluation": "validation_tuned_threshold",
                **test_result["tuned_metrics"],
            },
        ]
    )

    test_metrics.to_csv(
        reports_metrics_dir
        / "best_model_test_metrics.csv",
        index=False,
    )

    test_result["confusion_default"].to_csv(
        reports_metrics_dir
        / "best_model_confusion_default.csv",
    )
    test_result["confusion_tuned"].to_csv(
        reports_metrics_dir
        / "best_model_confusion_tuned.csv",
    )

    joblib.dump(
        best_model,
        models_dir / "best_model.joblib",
    )

    metadata = {
        "best_model": best_name,
        "selection_metric": PRIMARY_METRIC,
        "validation_pr_auc": float(
            validation_metrics.iloc[0][PRIMARY_METRIC]
        ),
        "threshold_selection": "best_validation_f1",
        "selected_threshold": float(
            test_result["threshold_result"].threshold
        ),
        "test_pr_auc": float(
            test_result["default_metrics"]["pr_auc"]
        ),
        "test_roc_auc": float(
            test_result["default_metrics"]["roc_auc"]
        ),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
    }

    with (
        models_dir / "best_model_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Validation ranking:")
    print(
        validation_metrics[
            [
                "model",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )

    print()
    print("Best model:", best_name)
    print(
        "Validation-selected threshold:",
        f"{test_result['threshold_result'].threshold:.4f}",
    )
    print()
    print("Best-model test metrics:")
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
