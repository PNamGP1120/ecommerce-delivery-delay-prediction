from __future__ import annotations

from collections import Counter
from threading import Lock


class PredictionTelemetry:
    def __init__(self) -> None:
        self._lock = Lock()

        self.total_requests = 0
        self.total_predictions = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.score_sum = 0.0
        self.review_count = 0
        self.band_counts: Counter[str] = Counter()
        self.missing_feature_values = 0

    def record_batch(
        self,
        *,
        scores: list[float],
        bands: list[str],
        requires_review: list[bool],
        latency_ms: float,
        missing_feature_values: int,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_predictions += len(
                scores
            )
            self.total_latency_ms += (
                latency_ms
            )
            self.score_sum += sum(
                scores
            )
            self.review_count += sum(
                bool(value)
                for value in requires_review
            )
            self.band_counts.update(
                bands
            )
            self.missing_feature_values += (
                missing_feature_values
            )

    def record_failure(self) -> None:
        with self._lock:
            self.failed_requests += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self.total_latency_ms
                / self.total_requests
                if self.total_requests
                else 0.0
            )

            mean_score = (
                self.score_sum
                / self.total_predictions
                if self.total_predictions
                else 0.0
            )

            action_rate = (
                self.review_count
                / self.total_predictions
                if self.total_predictions
                else 0.0
            )

            return {
                "total_requests": (
                    self.total_requests
                ),
                "total_predictions": (
                    self.total_predictions
                ),
                "failed_requests": (
                    self.failed_requests
                ),
                "average_request_latency_ms": (
                    avg_latency
                ),
                "mean_risk_score": (
                    mean_score
                ),
                "review_rate": (
                    action_rate
                ),
                "risk_band_counts": dict(
                    self.band_counts
                ),
                "missing_feature_values": (
                    self.missing_feature_values
                ),
            }
