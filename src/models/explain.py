from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, average_precision_score


def native_xgboost_importance(
    pipeline,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]
    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importances = np.asarray(
        model.feature_importances_,
        dtype="float64",
    )

    if len(feature_names) != len(importances):
        raise ValueError(
            "Transformed feature names and native "
            "importance lengths do not match."
        )

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def raw_permutation_importance(
    pipeline,
    X,
    y,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": (
                    result.importances_mean
                ),
                "importance_std": (
                    result.importances_std
                ),
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def shap_global_importance(
    pipeline,
    X,
    *,
    sample_size: int = 2_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, object, np.ndarray]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is optional for Phase 5. "
            "Install it with: uv add shap"
        ) from exc

    if len(X) > sample_size:
        X_sample = X.sample(
            n=sample_size,
            random_state=random_state,
        )
    else:
        X_sample = X.copy()

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]
    model = pipeline.named_steps[
        "model"
    ]

    transformed = preprocessor.transform(
        X_sample
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = np.asarray(
        preprocessor.get_feature_names_out()
    )

    explainer = shap.TreeExplainer(
        model
    )
    shap_values = explainer(
        transformed
    )

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        importance,
        shap_values,
        feature_names,
    )
