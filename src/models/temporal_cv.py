from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def make_expanding_window_folds(
    frame: pd.DataFrame,
    *,
    time_column: str = "prediction_timestamp",
    n_splits: int = 4,
    initial_train_fraction: float = 0.50,
) -> list[TemporalFold]:
    """Create expanding-window folds over a time-ordered event dataset.

    The first ``initial_train_fraction`` of observations forms the initial
    training window. The remaining observations are divided into equal-sized
    validation blocks. Each subsequent training set expands to include all
    prior validation blocks.

    Unlike TimeSeriesSplit, this helper makes the row-count design explicit
    for irregular event-time data while preserving strict chronology.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    if not (0.2 <= initial_train_fraction < 1.0):
        raise ValueError(
            "initial_train_fraction must be in [0.2, 1.0)."
        )

    if time_column not in frame.columns:
        raise ValueError(
            f"Missing time column: {time_column}"
        )

    ordered = (
        frame.sort_values(
            [time_column, "order_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    n_rows = len(ordered)
    initial_train_size = int(
        n_rows * initial_train_fraction
    )
    remaining = n_rows - initial_train_size

    if remaining < n_splits:
        raise ValueError(
            "Not enough rows for the requested number of splits."
        )

    block_sizes = np.full(
        n_splits,
        remaining // n_splits,
        dtype=int,
    )
    block_sizes[: remaining % n_splits] += 1

    folds: list[TemporalFold] = []
    validation_start_index = initial_train_size

    for fold_number, block_size in enumerate(
        block_sizes,
        start=1,
    ):
        validation_end_index = (
            validation_start_index + int(block_size)
        )

        train_index = np.arange(
            0,
            validation_start_index,
            dtype=int,
        )
        validation_index = np.arange(
            validation_start_index,
            validation_end_index,
            dtype=int,
        )

        train_times = ordered.loc[
            train_index,
            time_column,
        ]
        validation_times = ordered.loc[
            validation_index,
            time_column,
        ]

        assert train_times.max() <= validation_times.min()

        folds.append(
            TemporalFold(
                fold=fold_number,
                train_index=train_index,
                validation_index=validation_index,
                train_start=pd.Timestamp(
                    train_times.min()
                ),
                train_end=pd.Timestamp(
                    train_times.max()
                ),
                validation_start=pd.Timestamp(
                    validation_times.min()
                ),
                validation_end=pd.Timestamp(
                    validation_times.max()
                ),
            )
        )

        validation_start_index = validation_end_index

    assert validation_start_index == n_rows

    return folds


def fold_summary(
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    *,
    time_column: str = "prediction_timestamp",
    target_column: str = "late_delivery",
) -> pd.DataFrame:
    ordered = (
        frame.sort_values(
            [time_column, "order_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    rows = []

    for fold in folds:
        train = ordered.iloc[fold.train_index]
        validation = ordered.iloc[
            fold.validation_index
        ]

        rows.append(
            {
                "fold": fold.fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_late_rate": train[
                    target_column
                ].mean(),
                "validation_late_rate": validation[
                    target_column
                ].mean(),
            }
        )

    return pd.DataFrame(rows)
