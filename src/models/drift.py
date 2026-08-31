from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-6


def _psi_from_shares(
    reference_share: np.ndarray,
    comparison_share: np.ndarray,
) -> float:
    reference_share = np.clip(
        reference_share,
        EPSILON,
        None,
    )
    comparison_share = np.clip(
        comparison_share,
        EPSILON,
        None,
    )

    return float(
        np.sum(
            (comparison_share - reference_share)
            * np.log(
                comparison_share / reference_share
            )
        )
    )


def numeric_psi(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    bins: int = 10,
) -> float:
    ref = pd.to_numeric(
        reference,
        errors="coerce",
    )
    cmp = pd.to_numeric(
        comparison,
        errors="coerce",
    )

    valid_ref = ref.dropna()

    if (
        valid_ref.empty
        or valid_ref.nunique() < 2
    ):
        return 0.0

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )
    edges = np.unique(
        valid_ref.quantile(
            quantiles
        ).to_numpy()
    )

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_binned = pd.cut(
        ref,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    cmp_binned = pd.cut(
        cmp,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    categories = ref_binned.cat.categories

    ref_share = (
        ref_binned.value_counts(
            normalize=True,
            sort=False,
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )
    cmp_share = (
        cmp_binned.value_counts(
            normalize=True,
            sort=False,
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    ref_missing = float(ref.isna().mean())
    cmp_missing = float(cmp.isna().mean())

    ref_share = np.append(
        ref_share * (1 - ref_missing),
        ref_missing,
    )
    cmp_share = np.append(
        cmp_share * (1 - cmp_missing),
        cmp_missing,
    )

    return _psi_from_shares(
        ref_share,
        cmp_share,
    )


def categorical_psi(
    reference: pd.Series,
    comparison: pd.Series,
) -> float:
    ref = (
        reference.astype("object")
        .where(
            reference.notna(),
            "__MISSING__",
        )
        .astype(str)
    )
    cmp = (
        comparison.astype("object")
        .where(
            comparison.notna(),
            "__MISSING__",
        )
        .astype(str)
    )

    categories = sorted(
        set(ref.unique())
        | set(cmp.unique())
    )

    ref_share = (
        ref.value_counts(
            normalize=True,
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )
    cmp_share = (
        cmp.value_counts(
            normalize=True,
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    return _psi_from_shares(
        ref_share,
        cmp_share,
    )


def drift_summary(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in numeric_features:
        psi = numeric_psi(
            reference[feature],
            comparison[feature],
        )
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "psi": psi,
                "reference_missing_pct": (
                    reference[
                        feature
                    ].isna().mean()
                    * 100
                ),
                "comparison_missing_pct": (
                    comparison[
                        feature
                    ].isna().mean()
                    * 100
                ),
            }
        )

    for feature in categorical_features:
        psi = categorical_psi(
            reference[feature],
            comparison[feature],
        )
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "psi": psi,
                "reference_missing_pct": (
                    reference[
                        feature
                    ].isna().mean()
                    * 100
                ),
                "comparison_missing_pct": (
                    comparison[
                        feature
                    ].isna().mean()
                    * 100
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["drift_flag"] = pd.cut(
        result["psi"],
        bins=[
            -np.inf,
            0.10,
            0.25,
            np.inf,
        ],
        labels=[
            "low",
            "moderate",
            "high",
        ],
    )

    return (
        result.sort_values(
            "psi",
            ascending=False,
        )
        .reset_index(drop=True)
    )
