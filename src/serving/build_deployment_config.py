from __future__ import annotations

import json

import pandas as pd

from src.features.build_features import (
    find_project_root,
)
from src.serving.deployment import (
    build_deployment_config_data,
    write_deployment_config,
)


def main() -> None:
    project_root = find_project_root()

    model_path = (
        project_root
        / "models"
        / "best_tuned_candidate.joblib"
    )
    metadata_path = (
        project_root
        / "models"
        / "best_tuned_candidate_metadata.json"
    )
    oof_path = (
        project_root
        / "reports"
        / "metrics"
        / "best_tuned_oof_predictions.parquet"
    )
    thresholds_path = (
        project_root
        / "reports"
        / "metrics"
        / "tuned_operating_thresholds.csv"
    )
    output_path = (
        project_root
        / "models"
        / "deployment_config.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    oof = pd.read_parquet(
        oof_path
    )
    thresholds = pd.read_csv(
        thresholds_path
    )

    config = build_deployment_config_data(
        model_path=model_path,
        model_metadata=metadata,
        oof_predictions=oof,
        operating_thresholds=thresholds,
    )

    write_deployment_config(
        config,
        output_path,
    )

    print(
        "✓ Deployment config:",
        output_path,
    )
    print(
        "Candidate:",
        config["model"]["candidate"],
    )
    print(
        "Model SHA256:",
        config["model"]["sha256"],
    )
    print(
        "Action threshold:",
        (
            config["risk_score"][
                "action_threshold"
            ]
        ),
    )
    print(
        "Risk bands:",
        config["risk_score"]["bands"],
    )


if __name__ == "__main__":
    main()
