from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


RiskBand = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class FeaturePayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    purchase_year: int
    purchase_month: int = Field(
        ge=1,
        le=12,
    )
    purchase_weekday: int = Field(
        ge=0,
        le=6,
    )
    purchase_hour: int = Field(
        ge=0,
        le=23,
    )
    purchase_is_weekend: int = Field(
        ge=0,
        le=1,
    )

    purchase_month_sin: float
    purchase_month_cos: float
    purchase_weekday_sin: float
    purchase_weekday_cos: float
    purchase_hour_sin: float
    purchase_hour_cos: float

    approval_lag_hours: float = Field(
        ge=0,
    )
    promised_delivery_days: float = Field(
        ge=0,
    )

    item_count: int = Field(
        ge=1,
    )
    unique_products: int = Field(
        ge=1,
    )
    seller_count: int = Field(
        ge=1,
    )

    total_price: float = Field(
        ge=0,
    )
    total_freight: float = Field(
        ge=0,
    )
    order_total_value: float = Field(
        ge=0,
    )
    mean_item_price: float = Field(
        ge=0,
    )
    max_item_price: float = Field(
        ge=0,
    )
    freight_ratio: float = Field(
        ge=0,
        le=1,
    )

    payment_records: int | None = Field(
        default=None,
        ge=0,
    )
    payment_value: float | None = Field(
        default=None,
        ge=0,
    )
    max_installments: int | None = Field(
        default=None,
        ge=0,
    )
    payment_type_count: int | None = Field(
        default=None,
        ge=0,
    )

    mean_product_weight_g: float | None = Field(
        default=None,
        ge=0,
    )
    total_product_weight_g: float | None = Field(
        default=None,
        ge=0,
    )
    mean_product_volume_cm3: float | None = Field(
        default=None,
        ge=0,
    )
    total_product_volume_cm3: float | None = Field(
        default=None,
        ge=0,
    )
    category_count: int = Field(
        ge=1,
    )
    mean_product_photos: float | None = Field(
        default=None,
        ge=0,
    )

    seller_state_count: int = Field(
        ge=1,
    )
    mean_distance_km: float | None = Field(
        default=None,
        ge=0,
    )
    max_distance_km: float | None = Field(
        default=None,
        ge=0,
    )
    min_distance_km: float | None = Field(
        default=None,
        ge=0,
    )
    all_sellers_same_state: int = Field(
        ge=0,
        le=1,
    )
    same_state_seller_share: float = Field(
        ge=0,
        le=1,
    )

    customer_state: str = Field(
        min_length=2,
        max_length=2,
    )
    primary_payment_type: str | None = None
    dominant_product_category: str = Field(
        min_length=1,
    )

    @field_validator("customer_state")
    @classmethod
    def normalize_state(
        cls,
        value: str,
    ) -> str:
        return value.strip().upper()

    @field_validator(
        "primary_payment_type",
        "dominant_product_category",
    )
    @classmethod
    def normalize_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    order_id: str | None = None
    prediction_timestamp: datetime | None = None
    features: FeaturePayload


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    items: list[PredictionRequest] = Field(
        min_length=1,
        max_length=500,
    )


class PredictionResponse(BaseModel):
    order_id: str | None
    request_id: str
    risk_score: float = Field(
        ge=0,
        le=1,
    )
    risk_band: RiskBand
    requires_review: bool
    action_threshold: float = Field(
        ge=0,
        le=1,
    )
    calibrated_probability: bool = False
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    app_version: str
    candidate: str
    model_family: str
    model_version: str
    feature_count: int
    risk_score_semantics: str
    calibration_status: str
    action_threshold: float
    action_threshold_rule: str
    risk_band_source: str
    warning: str
