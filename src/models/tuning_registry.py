from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor
from src.models.model_registry import (
    RANDOM_STATE,
    _build_dense_preprocessor,
)


@dataclass(frozen=True)
class TuningConfig:
    config_id: str
    model_family: str
    params: dict[str, Any]


def _grid(
    model_family: str,
    grid: dict[str, list[Any]],
) -> list[TuningConfig]:
    keys = list(grid)
    values = [grid[key] for key in keys]

    configs = []
    for index, combination in enumerate(
        product(*values),
        start=1,
    ):
        params = dict(zip(keys, combination))
        configs.append(
            TuningConfig(
                config_id=f"{model_family}_{index:02d}",
                model_family=model_family,
                params=params,
            )
        )

    return configs


def get_tuning_configs(
    *,
    quick: bool = False,
) -> list[TuningConfig]:
    if quick:
        return [
            TuningConfig(
                config_id="logistic_regression_quick",
                model_family="logistic_regression",
                params={
                    "C": 1.0,
                    "class_weight": "balanced",
                },
            ),
            TuningConfig(
                config_id="random_forest_quick",
                model_family="random_forest",
                params={
                    "n_estimators": 250,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "max_features": "sqrt",
                },
            ),
            TuningConfig(
                config_id="hist_gradient_boosting_quick",
                model_family="hist_gradient_boosting",
                params={
                    "learning_rate": 0.08,
                    "max_leaf_nodes": 31,
                    "min_samples_leaf": 30,
                },
            ),
            TuningConfig(
                config_id="xgboost_quick",
                model_family="xgboost",
                params={
                    "n_estimators": 350,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "min_child_weight": 5,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 2.0,
                    "scale_pos_weight": "auto",
                },
            ),
        ]

    configs: list[TuningConfig] = []

    configs.extend(
        _grid(
            "logistic_regression",
            {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [
                    None,
                    "balanced",
                ],
            },
        )
    )

    configs.extend(
        _grid(
            "random_forest",
            {
                "n_estimators": [300],
                "max_depth": [12, None],
                "min_samples_leaf": [5, 20],
                "max_features": ["sqrt"],
            },
        )
    )

    configs.extend(
        _grid(
            "hist_gradient_boosting",
            {
                "learning_rate": [0.05, 0.10],
                "max_leaf_nodes": [15, 31],
                "min_samples_leaf": [30],
            },
        )
    )

    configs.extend(
        _grid(
            "xgboost",
            {
                "n_estimators": [300, 500],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.07],
                "min_child_weight": [5],
                "subsample": [0.85],
                "colsample_bytree": [0.85],
                "reg_lambda": [2.0],
                "scale_pos_weight": ["auto"],
            },
        )
    )

    return configs


def build_pipeline(
    config: TuningConfig,
    *,
    y_train=None,
) -> Pipeline:
    params = dict(config.params)

    if config.model_family == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            **params,
        )
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

    if config.model_family == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params,
        )
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

    if config.model_family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=220,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            **params,
        )
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    _build_dense_preprocessor(),
                ),
                ("model", estimator),
            ]
        )

    if config.model_family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for Phase 4B. "
                "Install it with: uv add xgboost"
            ) from exc

        scale_pos_weight = params.pop(
            "scale_pos_weight",
            1.0,
        )

        if scale_pos_weight == "auto":
            if y_train is None:
                raise ValueError(
                    "y_train is required for automatic "
                    "scale_pos_weight."
                )

            positives = float(y_train.sum())
            negatives = float(len(y_train) - positives)

            scale_pos_weight = (
                negatives / positives
                if positives > 0
                else 1.0
            )

        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            **params,
        )

        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

    raise ValueError(
        f"Unknown model family: {config.model_family}"
    )
