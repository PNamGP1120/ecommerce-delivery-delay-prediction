from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from app.config import Settings
from app.logging_config import (
    configure_logging,
)
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.monitoring.telemetry import (
    PredictionTelemetry,
)
from src.serving.model_service import (
    ModelService,
)


logger = logging.getLogger(
    "delivery_delay.api"
)


def create_app(
    service: ModelService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = (
        settings
        or Settings.from_env()
    )

    configure_logging(
        settings.log_level
    )

    telemetry = (
        service.telemetry
        if service is not None
        else PredictionTelemetry()
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        if service is not None:
            app.state.model_service = (
                service
            )
        else:
            app.state.model_service = (
                ModelService.from_paths(
                    model_path=(
                        settings.model_path
                    ),
                    model_metadata_path=(
                        settings.model_metadata_path
                    ),
                    deployment_config_path=(
                        settings.deployment_config_path
                    ),
                    telemetry=telemetry,
                    action_threshold_override=(
                        settings.action_threshold_override
                    ),
                )
            )

        logger.info(
            "model_loaded",
            extra={
                "event": "model_loaded",
                "model_version": (
                    app.state.model_service.model_version
                ),
            },
        )

        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Predicts a delivery-delay risk score at "
            "order approval time. The returned risk_score "
            "is a ranking score, not a calibrated probability."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next,
    ):
        request_id = (
            request.headers.get(
                "x-request-id"
            )
            or str(uuid.uuid4())
        )
        request.state.request_id = (
            request_id
        )

        started = time.perf_counter()

        try:
            response = await call_next(
                request
            )
        except Exception:
            telemetry.record_failure()

            logger.exception(
                "request_failed",
                extra={
                    "event": "request_failed",
                    "request_id": (
                        request_id
                    ),
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000

        response.headers[
            "x-request-id"
        ] = request_id

        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": (
                    response.status_code
                ),
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
            },
        )

        return response

    def get_service(
        request: Request,
    ) -> ModelService:
        return request.app.state.model_service

    @app.get(
        "/",
        tags=["system"],
    )
    def root():
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health(
        request: Request,
    ):
        model_service = get_service(
            request
        )

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=(
                model_service.model_version
            ),
        )

    @app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        tags=["system"],
    )
    def model_info(
        request: Request,
    ):
        model_service = get_service(
            request
        )

        risk_config = (
            model_service.deployment_config[
                "risk_score"
            ]
        )

        return ModelInfoResponse(
            app_version=(
                settings.app_version
            ),
            candidate=(
                model_service.candidate
            ),
            model_family=(
                model_service.model_family
            ),
            model_version=(
                model_service.model_version
            ),
            feature_count=(
                model_service.feature_count
            ),
            risk_score_semantics=(
                risk_config[
                    "semantics"
                ]
            ),
            calibration_status=(
                "not_calibrated"
            ),
            action_threshold=(
                model_service.action_threshold
            ),
            action_threshold_rule=(
                risk_config[
                    "action_threshold_rule"
                ]
            ),
            risk_band_source=(
                risk_config[
                    "risk_band_source"
                ]
            ),
            warning=(
                risk_config[
                    "warning"
                ]
            ),
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
    )
    def predict(
        payload: PredictionRequest,
        request: Request,
    ):
        model_service = get_service(
            request
        )
        request_id = (
            request.state.request_id
        )

        try:
            result = model_service.predict(
                [
                    payload.features.model_dump()
                ],
                request_id=request_id,
            )[0]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return PredictionResponse(
            order_id=payload.order_id,
            request_id=request_id,
            risk_score=(
                result.risk_score
            ),
            risk_band=(
                result.risk_band
            ),
            requires_review=(
                result.requires_review
            ),
            action_threshold=(
                model_service.action_threshold
            ),
            calibrated_probability=False,
            model_version=(
                model_service.model_version
            ),
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["prediction"],
    )
    def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ):
        if (
            len(payload.items)
            > settings.max_batch_size
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Batch exceeds configured "
                    f"maximum size of "
                    f"{settings.max_batch_size}."
                ),
            )

        model_service = get_service(
            request
        )
        request_id = (
            request.state.request_id
        )

        try:
            results = model_service.predict(
                [
                    item.features.model_dump()
                    for item in payload.items
                ],
                request_id=request_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        predictions = []

        for index, (
            item,
            result,
        ) in enumerate(
            zip(
                payload.items,
                results,
                strict=True,
            )
        ):
            predictions.append(
                PredictionResponse(
                    order_id=item.order_id,
                    request_id=(
                        f"{request_id}:{index}"
                    ),
                    risk_score=(
                        result.risk_score
                    ),
                    risk_band=(
                        result.risk_band
                    ),
                    requires_review=(
                        result.requires_review
                    ),
                    action_threshold=(
                        model_service.action_threshold
                    ),
                    calibrated_probability=False,
                    model_version=(
                        model_service.model_version
                    ),
                )
            )

        return BatchPredictionResponse(
            request_id=request_id,
            predictions=predictions,
        )

    @app.get(
        "/monitoring/snapshot",
        tags=["monitoring"],
    )
    def monitoring_snapshot(
        request: Request,
    ):
        model_service = get_service(
            request
        )

        return {
            "model_version": (
                model_service.model_version
            ),
            "reference": (
                model_service.deployment_config[
                    "monitoring_reference"
                ]
            ),
            "runtime": (
                model_service.telemetry.snapshot()
            ),
            "notes": {
                "risk_score": (
                    "uncalibrated ranking score"
                ),
                "monitor": [
                    "mean_risk_score",
                    "review_rate",
                    "risk_band_counts",
                    "missing_feature_values",
                    "request latency",
                ],
            },
        }

    return app


app = create_app()
