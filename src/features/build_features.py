from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "late_delivery"
ID_COLUMNS = ["order_id", "prediction_timestamp"]

FORBIDDEN_MODEL_FEATURES = {
    "customer_id",
    "customer_unique_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "delivery_delay_days",
    "review_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
}

NUMERIC_FEATURES = [
    "purchase_year",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "purchase_is_weekend",
    "purchase_month_sin",
    "purchase_month_cos",
    "purchase_weekday_sin",
    "purchase_weekday_cos",
    "purchase_hour_sin",
    "purchase_hour_cos",
    "approval_lag_hours",
    "promised_delivery_days",
    "item_count",
    "unique_products",
    "seller_count",
    "total_price",
    "total_freight",
    "order_total_value",
    "mean_item_price",
    "max_item_price",
    "freight_ratio",
    "payment_records",
    "payment_value",
    "max_installments",
    "payment_type_count",
    "mean_product_weight_g",
    "total_product_weight_g",
    "mean_product_volume_cm3",
    "total_product_volume_cm3",
    "category_count",
    "mean_product_photos",
    "seller_state_count",
    "mean_distance_km",
    "max_distance_km",
    "min_distance_km",
    "all_sellers_same_state",
    "same_state_seller_share",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "primary_payment_type",
    "dominant_product_category",
]

MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data" / "raw").exists():
            return candidate

    raise FileNotFoundError("Không tìm thấy project root chứa data/raw.")


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    result = numerator.astype("float64") / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    lat1_rad = np.radians(lat1.astype("float64"))
    lon1_rad = np.radians(lon1.astype("float64"))
    lat2_rad = np.radians(lat2.astype("float64"))
    lon2_rad = np.radians(lon2.astype("float64"))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2) ** 2
    )

    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def build_geolocation_lookup(
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    }
    missing = required.difference(geolocation.columns)
    if missing:
        raise ValueError(f"Missing geolocation columns: {sorted(missing)}")

    lookup = (
        geolocation[
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "geolocation_zip_code_prefix",
            as_index=False,
        )
        .agg(
            latitude=("geolocation_lat", "median"),
            longitude=("geolocation_lng", "median"),
        )
    )

    assert lookup["geolocation_zip_code_prefix"].is_unique
    return lookup


def aggregate_order_items(
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    }
    missing = required.difference(order_items.columns)
    if missing:
        raise ValueError(f"Missing order-item columns: {sorted(missing)}")

    result = (
        order_items.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            mean_item_price=("price", "mean"),
            max_item_price=("price", "max"),
        )
    )

    result["order_total_value"] = (
        result["total_price"] + result["total_freight"]
    )
    result["freight_ratio"] = _safe_divide(
        result["total_freight"],
        result["order_total_value"],
    )

    assert result["order_id"].is_unique
    return result


def aggregate_payments(
    payments: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    }
    missing = required.difference(payments.columns)
    if missing:
        raise ValueError(f"Missing payment columns: {sorted(missing)}")

    payment_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_records=("payment_sequential", "count"),
            payment_value=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
            payment_type_count=("payment_type", "nunique"),
        )
    )

    primary_payment = (
        payments.sort_values(
            ["order_id", "payment_sequential"],
            kind="stable",
        )
        .drop_duplicates("order_id", keep="first")
        [["order_id", "payment_type"]]
        .rename(
            columns={
                "payment_type": "primary_payment_type",
            }
        )
    )

    result = payment_agg.merge(
        primary_payment,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    assert result["order_id"].is_unique
    return result


def aggregate_product_features(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    product_cols = {
        "product_id",
        "product_category_name",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    }
    missing = product_cols.difference(products.columns)
    if missing:
        raise ValueError(f"Missing product columns: {sorted(missing)}")

    products_enriched = products[
        [
            "product_id",
            "product_category_name",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ].copy()

    products_enriched = products_enriched.merge(
        category_translation,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    products_enriched["product_volume_cm3"] = (
        products_enriched["product_length_cm"]
        * products_enriched["product_height_cm"]
        * products_enriched["product_width_cm"]
    )

    item_products = order_items[
        ["order_id", "product_id"]
    ].merge(
        products_enriched,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    item_products["product_weight_g"] = pd.to_numeric(
        item_products["product_weight_g"],
        errors="coerce",
    )
    item_products["product_volume_cm3"] = pd.to_numeric(
        item_products["product_volume_cm3"],
        errors="coerce",
    )

    item_products["category_for_count"] = (
        item_products["product_category_name_english"]
        .fillna("unknown")
    )

    product_agg = (
        item_products.groupby("order_id", as_index=False)
        .agg(
            mean_product_weight_g=("product_weight_g", "mean"),
            total_product_weight_g=("product_weight_g", "sum"),
            mean_product_volume_cm3=("product_volume_cm3", "mean"),
            total_product_volume_cm3=("product_volume_cm3", "sum"),
            category_count=("category_for_count", "nunique"),
            mean_product_photos=("product_photos_qty", "mean"),
        )
    )

    category_counts = (
        item_products.groupby(
            ["order_id", "category_for_count"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "category_items"})
        .sort_values(
            ["order_id", "category_items", "category_for_count"],
            ascending=[True, False, True],
            kind="stable",
        )
    )

    dominant_category = (
        category_counts
        .drop_duplicates("order_id", keep="first")
        [["order_id", "category_for_count"]]
        .rename(
            columns={
                "category_for_count": "dominant_product_category",
            }
        )
    )

    result = product_agg.merge(
        dominant_category,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    assert result["order_id"].is_unique
    return result


def build_geographic_features(
    base_orders: pd.DataFrame,
    customers: pd.DataFrame,
    order_items: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo_lookup = build_geolocation_lookup(geolocation)

    customer_lookup = customers[
        [
            "customer_id",
            "customer_zip_code_prefix",
            "customer_state",
        ]
    ].copy()

    customer_geo = customer_lookup.merge(
        geo_lookup.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "latitude": "customer_latitude",
                "longitude": "customer_longitude",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )

    order_customer = base_orders[
        ["order_id", "customer_id"]
    ].merge(
        customer_geo,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    customer_features = order_customer[
        ["order_id", "customer_state"]
    ].copy()

    seller_geo = sellers[
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_state",
        ]
    ].merge(
        geo_lookup.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "latitude": "seller_latitude",
                "longitude": "seller_longitude",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )

    distance_rows = (
        order_items[["order_id", "seller_id"]]
        .merge(
            order_customer[
                [
                    "order_id",
                    "customer_state",
                    "customer_latitude",
                    "customer_longitude",
                ]
            ],
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            seller_geo,
            on="seller_id",
            how="left",
            validate="many_to_one",
        )
    )

    distance_rows["seller_customer_distance_km"] = _haversine_km(
        distance_rows["customer_latitude"],
        distance_rows["customer_longitude"],
        distance_rows["seller_latitude"],
        distance_rows["seller_longitude"],
    )

    distance_rows["seller_same_state"] = (
        distance_rows["customer_state"]
        == distance_rows["seller_state"]
    ).astype("int8")

    distance_features = (
        distance_rows.groupby("order_id", as_index=False)
        .agg(
            seller_state_count=("seller_state", "nunique"),
            mean_distance_km=("seller_customer_distance_km", "mean"),
            max_distance_km=("seller_customer_distance_km", "max"),
            min_distance_km=("seller_customer_distance_km", "min"),
            all_sellers_same_state=("seller_same_state", "min"),
            same_state_seller_share=("seller_same_state", "mean"),
        )
    )

    assert customer_features["order_id"].is_unique
    assert distance_features["order_id"].is_unique

    return customer_features, distance_features


def build_order_base_features(
    orders_cleaned: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
        TARGET_COLUMN,
    }
    missing = required.difference(orders_cleaned.columns)
    if missing:
        raise ValueError(f"Missing cleaned-order columns: {sorted(missing)}")

    base = orders_cleaned[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
            TARGET_COLUMN,
        ]
    ].copy()

    purchase_ts = pd.to_datetime(
        base["order_purchase_timestamp"],
        errors="coerce",
    )
    approval_ts = pd.to_datetime(
        base["order_approved_at"],
        errors="coerce",
    )
    estimated_ts = pd.to_datetime(
        base["order_estimated_delivery_date"],
        errors="coerce",
    )

    base["prediction_timestamp"] = approval_ts

    base["purchase_year"] = purchase_ts.dt.year.astype("int16")
    base["purchase_month"] = purchase_ts.dt.month.astype("int8")
    base["purchase_weekday"] = purchase_ts.dt.dayofweek.astype("int8")
    base["purchase_hour"] = purchase_ts.dt.hour.astype("int8")
    base["purchase_is_weekend"] = (
        base["purchase_weekday"].isin([5, 6]).astype("int8")
    )

    base["purchase_month_sin"] = np.sin(
        2 * np.pi * (base["purchase_month"] - 1) / 12
    )
    base["purchase_month_cos"] = np.cos(
        2 * np.pi * (base["purchase_month"] - 1) / 12
    )
    base["purchase_weekday_sin"] = np.sin(
        2 * np.pi * base["purchase_weekday"] / 7
    )
    base["purchase_weekday_cos"] = np.cos(
        2 * np.pi * base["purchase_weekday"] / 7
    )
    base["purchase_hour_sin"] = np.sin(
        2 * np.pi * base["purchase_hour"] / 24
    )
    base["purchase_hour_cos"] = np.cos(
        2 * np.pi * base["purchase_hour"] / 24
    )

    base["approval_lag_hours"] = (
        approval_ts - purchase_ts
    ).dt.total_seconds() / 3600

    base["promised_delivery_days"] = (
        estimated_ts - approval_ts
    ).dt.total_seconds() / 86_400

    return base


def validate_feature_dataset(
    features: pd.DataFrame,
    expected_order_count: int | None = None,
) -> None:
    if expected_order_count is not None:
        assert len(features) == expected_order_count

    assert features["order_id"].is_unique
    assert features[TARGET_COLUMN].isin([0, 1]).all()
    assert features["prediction_timestamp"].notna().all()

    assert (
        features["promised_delivery_days"].dropna() >= 0
    ).all()

    missing_features = set(MODEL_FEATURE_COLUMNS).difference(
        features.columns
    )
    assert not missing_features, (
        f"Missing model features: {sorted(missing_features)}"
    )

    forbidden_present = FORBIDDEN_MODEL_FEATURES.intersection(
        MODEL_FEATURE_COLUMNS
    )
    assert not forbidden_present, (
        f"Forbidden features registered in X: "
        f"{sorted(forbidden_present)}"
    )

    actual_forbidden_columns = (
        FORBIDDEN_MODEL_FEATURES.intersection(features.columns)
        - {"customer_id"}
    )
    assert not actual_forbidden_columns, (
        "Forbidden columns leaked into processed feature dataset: "
        f"{sorted(actual_forbidden_columns)}"
    )


def build_feature_dataset(
    orders_cleaned: pd.DataFrame,
    customers: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    base = build_order_base_features(orders_cleaned)

    item_features = aggregate_order_items(order_items)
    payment_features = aggregate_payments(payments)
    product_features = aggregate_product_features(
        order_items,
        products,
        category_translation,
    )
    customer_features, geographic_features = build_geographic_features(
        base_orders=orders_cleaned,
        customers=customers,
        order_items=order_items,
        sellers=sellers,
        geolocation=geolocation,
    )

    features = base.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        product_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        customer_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        geographic_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    features = features.drop(
        columns=[
            "customer_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
        ],
        errors="ignore",
    )

    ordered_columns = (
        ["order_id", "prediction_timestamp"]
        + MODEL_FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )
    features = features[ordered_columns].copy()

    validate_feature_dataset(
        features,
        expected_order_count=len(orders_cleaned),
    )

    return features


def make_chronological_splits(
    features: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplit:
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1.")

    if not (0 < validation_fraction < 1):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be < 1."
        )

    ordered = (
        features.sort_values(
            ["prediction_timestamp", "order_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = int(
        n * (train_fraction + validation_fraction)
    )

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[
        train_end:validation_end
    ].copy()
    test = ordered.iloc[validation_end:].copy()

    assert len(train) + len(validation) + len(test) == n
    assert set(train["order_id"]).isdisjoint(validation["order_id"])
    assert set(train["order_id"]).isdisjoint(test["order_id"])
    assert set(validation["order_id"]).isdisjoint(test["order_id"])

    assert (
        train["prediction_timestamp"].max()
        <= validation["prediction_timestamp"].min()
    )
    assert (
        validation["prediction_timestamp"].max()
        <= test["prediction_timestamp"].min()
    )

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )


def load_project_tables(
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"

    return {
        "orders_cleaned": pd.read_parquet(
            interim_dir / "orders_cleaned.parquet"
        ),
        "customers": pd.read_csv(
            raw_dir / "olist_customers_dataset.csv"
        ),
        "order_items": pd.read_csv(
            raw_dir / "olist_order_items_dataset.csv"
        ),
        "payments": pd.read_csv(
            raw_dir / "olist_order_payments_dataset.csv"
        ),
        "products": pd.read_csv(
            raw_dir / "olist_products_dataset.csv"
        ),
        "sellers": pd.read_csv(
            raw_dir / "olist_sellers_dataset.csv"
        ),
        "geolocation": pd.read_csv(
            raw_dir / "olist_geolocation_dataset.csv"
        ),
        "category_translation": pd.read_csv(
            raw_dir / "product_category_name_translation.csv"
        ),
    }


def main() -> None:
    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "reports" / "metrics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tables = load_project_tables(project_root)

    features = build_feature_dataset(**tables)

    split = make_chronological_splits(features)

    features.to_parquet(
        processed_dir / "features.parquet",
        index=False,
    )
    split.train.to_parquet(
        processed_dir / "train.parquet",
        index=False,
    )
    split.validation.to_parquet(
        processed_dir / "validation.parquet",
        index=False,
    )
    split.test.to_parquet(
        processed_dir / "test.parquet",
        index=False,
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(split.train),
                "start": split.train["prediction_timestamp"].min(),
                "end": split.train["prediction_timestamp"].max(),
                "late_rate": split.train[TARGET_COLUMN].mean(),
            },
            {
                "split": "validation",
                "rows": len(split.validation),
                "start": split.validation["prediction_timestamp"].min(),
                "end": split.validation["prediction_timestamp"].max(),
                "late_rate": split.validation[TARGET_COLUMN].mean(),
            },
            {
                "split": "test",
                "rows": len(split.test),
                "start": split.test["prediction_timestamp"].min(),
                "end": split.test["prediction_timestamp"].max(),
                "late_rate": split.test[TARGET_COLUMN].mean(),
            },
        ]
    )

    split_summary.to_csv(
        metrics_dir / "feature_split_summary.csv",
        index=False,
    )

    print(
        f"✓ features.parquet: {len(features):,} rows, "
        f"{features.shape[1]} columns"
    )
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
