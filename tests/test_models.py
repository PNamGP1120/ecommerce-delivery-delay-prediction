import numpy as np
import pandas as pd

from src.models.evaluate import (
    classification_metrics,
    confusion_matrix_frame,
    find_best_f1_threshold,
)
from src.models.model_registry import get_model_registry


def test_model_registry_contains_required_baselines():
    registry = get_model_registry()

    expected = {
        "dummy_prior",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    }

    assert expected.issubset(registry)


def test_classification_metrics_binary_output():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    metrics = classification_metrics(
        y_true,
        y_probability,
        threshold=0.50,
    )

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_find_best_f1_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probability = np.array(
        [0.05, 0.10, 0.40, 0.45, 0.70, 0.90]
    )

    result = find_best_f1_threshold(
        y_true,
        y_probability,
    )

    assert 0.0 <= result.threshold <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_confusion_matrix_frame_shape():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array(
        [0.10, 0.70, 0.80, 0.20]
    )

    frame = confusion_matrix_frame(
        y_true,
        y_probability,
        threshold=0.50,
    )

    assert frame.shape == (2, 2)
    assert int(frame.to_numpy().sum()) == len(y_true)
