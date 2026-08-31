from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    pipeline_factory: Callable[[], Pipeline]


def _build_dense_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _standard_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _dense_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_dense_preprocessor()),
            ("model", estimator),
        ]
    )


def get_model_registry() -> dict[str, ModelSpec]:
    return {
        "dummy_prior": ModelSpec(
            name="dummy_prior",
            family="baseline",
            description=(
                "DummyClassifier using the training class prior."
            ),
            pipeline_factory=lambda: _standard_pipeline(
                DummyClassifier(
                    strategy="prior",
                )
            ),
        ),
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            family="linear",
            description=(
                "Class-balanced logistic regression baseline."
            ),
            pipeline_factory=lambda: _standard_pipeline(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            family="tree",
            description=(
                "Regularized class-balanced decision tree."
            ),
            pipeline_factory=lambda: _standard_pipeline(
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=50,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            family="ensemble",
            description=(
                "Random forest with balanced subsampling."
            ),
            pipeline_factory=lambda: _standard_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            family="boosting",
            description=(
                "Histogram gradient boosting with dense one-hot "
                "preprocessing."
            ),
            pipeline_factory=lambda: _dense_pipeline(
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=180,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=15,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
    }
