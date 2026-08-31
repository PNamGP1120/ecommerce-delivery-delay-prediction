from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "E-commerce Delivery Delay Risk API"
    app_version: str = "0.1.0"
    model_path: Path = Path(
        "models/best_tuned_candidate.joblib"
    )
    model_metadata_path: Path = Path(
        "models/best_tuned_candidate_metadata.json"
    )
    deployment_config_path: Path = Path(
        "models/deployment_config.json"
    )
    log_level: str = "INFO"
    max_batch_size: int = 500
    action_threshold_override: float | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        raw_threshold = os.getenv(
            "RISK_ACTION_THRESHOLD"
        )

        threshold = (
            float(raw_threshold)
            if raw_threshold not in {None, ""}
            else None
        )

        return cls(
            app_name=os.getenv(
                "APP_NAME",
                cls.app_name,
            ),
            app_version=os.getenv(
                "APP_VERSION",
                cls.app_version,
            ),
            model_path=Path(
                os.getenv(
                    "MODEL_PATH",
                    str(cls.model_path),
                )
            ),
            model_metadata_path=Path(
                os.getenv(
                    "MODEL_METADATA_PATH",
                    str(cls.model_metadata_path),
                )
            ),
            deployment_config_path=Path(
                os.getenv(
                    "DEPLOYMENT_CONFIG_PATH",
                    str(cls.deployment_config_path),
                )
            ),
            log_level=os.getenv(
                "LOG_LEVEL",
                cls.log_level,
            ).upper(),
            max_batch_size=int(
                os.getenv(
                    "MAX_BATCH_SIZE",
                    str(cls.max_batch_size),
                )
            ),
            action_threshold_override=threshold,
        )
