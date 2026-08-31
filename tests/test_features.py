from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import (
    FORBIDDEN_MODEL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_chronological_splits,
    validate_feature_dataset,
)


def _toy_features() -> pd.DataFrame:
    n = 20
    frame = pd.DataFrame(
        {
            "order_id": [f"order_{i:02d}" for i in range(n)],
            "prediction_timestamp": pd.date_range(
                "2024-01-01",
                periods=n,
                freq="D",
            ),
            TARGET_COLUMN: [0, 1] * 10,
        }
    )

    for col in MODEL_FEATURE_COLUMNS:
        if col in {
            "customer_state",
            "primary_payment_type",
            "dominant_product_category",
        }:
            frame[col] = "sample"
        else:
            frame[col] = 1.0

    frame["promised_delivery_days"] = 10.0
    return frame


def test_forbidden_features_not_registered_in_x():
    assert not (
        FORBIDDEN_MODEL_FEATURES
        & set(MODEL_FEATURE_COLUMNS)
    )


def test_validate_feature_dataset_preserves_grain():
    frame = _toy_features()

    validate_feature_dataset(
        frame,
        expected_order_count=len(frame),
    )


def test_chronological_split_has_no_overlap():
    frame = _toy_features()

    split = make_chronological_splits(
        frame,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3

    assert (
        split.train["prediction_timestamp"].max()
        <= split.validation["prediction_timestamp"].min()
    )
    assert (
        split.validation["prediction_timestamp"].max()
        <= split.test["prediction_timestamp"].min()
    )


def test_negative_promised_window_fails():
    frame = _toy_features()
    frame.loc[0, "promised_delivery_days"] = -1

    with pytest.raises(AssertionError):
        validate_feature_dataset(frame)
