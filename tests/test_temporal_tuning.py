import numpy as np
import pandas as pd

from src.models.temporal_cv import (
    make_expanding_window_folds,
)
from src.models.thresholds import (
    threshold_for_minimum_recall,
)
from src.models.tuning_registry import (
    get_tuning_configs,
)


def _toy_frame(n=100):
    return pd.DataFrame(
        {
            "order_id": [
                f"order_{i:03d}"
                for i in range(n)
            ],
            "prediction_timestamp": pd.date_range(
                "2024-01-01",
                periods=n,
                freq="h",
            ),
            "late_delivery": (
                np.arange(n) % 7 == 0
            ).astype("int8"),
        }
    )


def test_expanding_folds_are_strictly_chronological():
    frame = _toy_frame()

    folds = make_expanding_window_folds(
        frame,
        n_splits=4,
        initial_train_fraction=0.50,
    )

    assert len(folds) == 4

    for fold in folds:
        assert (
            fold.train_end
            <= fold.validation_start
        )
        assert set(
            fold.train_index
        ).isdisjoint(
            fold.validation_index
        )


def test_expanding_training_window_grows():
    frame = _toy_frame()

    folds = make_expanding_window_folds(
        frame,
        n_splits=4,
        initial_train_fraction=0.50,
    )

    train_sizes = [
        len(fold.train_index)
        for fold in folds
    ]

    assert train_sizes == sorted(
        train_sizes
    )
    assert len(set(train_sizes)) == 4


def test_registry_contains_xgboost():
    families = {
        config.model_family
        for config in get_tuning_configs(
            quick=True
        )
    }

    assert "xgboost" in families
    assert "random_forest" in families
    assert "logistic_regression" in families
    assert "hist_gradient_boosting" in families


def test_threshold_for_minimum_recall():
    y_true = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )
    probability = np.array(
        [0.05, 0.10, 0.25, 0.40,
         0.30, 0.50, 0.70, 0.90]
    )

    result = threshold_for_minimum_recall(
        y_true,
        probability,
        minimum_recall=0.50,
    )

    assert result.recall >= 0.50
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.threshold <= 1.0
