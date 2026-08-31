PYTHON := uv run python
PYTEST := uv run pytest
FASTAPI := uv run fastapi

IMAGE_NAME := delivery-delay-api
CONTAINER_NAME := delivery-delay-api
PORT := 8000

.PHONY: help install sync test test-features test-models test-tuning test-analysis test-serving \
        features baseline tune-quick tune evaluate deployment-config pipeline \
        api api-prod docker-build docker-run docker-stop docker-compose \
        clean clean-cache clean-models clean-reports all

help:
	@echo "E-commerce Delivery Delay Prediction"
	@echo ""
	@echo "Environment"
	@echo "  make install              Install/sync dependencies"
	@echo "  make sync                 Sync environment from uv.lock"
	@echo ""
	@echo "Pipeline"
	@echo "  make features             Build processed feature datasets"
	@echo "  make baseline             Train Phase 4A baseline models"
	@echo "  make tune-quick           Run Phase 4B quick temporal CV"
	@echo "  make tune                 Run full Phase 4B tuning"
	@echo "  make evaluate             Run Phase 5 evaluation"
	@echo "  make deployment-config    Build deployment metadata"
	@echo "  make pipeline             Run complete ML pipeline"
	@echo ""
	@echo "Testing"
	@echo "  make test                 Run all tests"
	@echo "  make test-features        Run feature tests"
	@echo "  make test-models          Run model tests"
	@echo "  make test-tuning          Run temporal tuning tests"
	@echo "  make test-analysis        Run evaluation tests"
	@echo "  make test-serving         Run API/serving tests"
	@echo ""
	@echo "API"
	@echo "  make api                  Start FastAPI development server"
	@echo "  make api-prod             Start production-style API server"
	@echo ""
	@echo "Docker"
	@echo "  make docker-build         Build Docker image"
	@echo "  make docker-run           Run Docker container"
	@echo "  make docker-stop          Stop Docker container"
	@echo "  make docker-compose       Run with Docker Compose"
	@echo ""
	@echo "Cleanup"
	@echo "  make clean                Remove Python/cache artifacts"
	@echo "  make clean-models         Remove generated model artifacts"
	@echo "  make clean-reports        Remove generated Phase 4-6 reports"
	@echo ""
	@echo "  make all                  Sync, test, pipeline, deployment config"

install:
	uv sync

sync:
	uv sync --locked

features:
	$(PYTHON) -m src.features.build_features

baseline:
	$(PYTHON) -m src.models.train

tune-quick:
	$(PYTHON) -m src.models.tune --quick

tune:
	$(PYTHON) -m src.models.tune

evaluate:
	$(PYTHON) -m src.models.analyze

deployment-config:
	$(PYTHON) -m src.serving.build_deployment_config

pipeline: features baseline tune evaluate deployment-config
	@echo ""
	@echo "✓ Complete ML pipeline finished."

test:
	$(PYTEST) -q

test-features:
	$(PYTEST) tests/test_features.py -q

test-models:
	$(PYTEST) tests/test_models.py -q

test-tuning:
	$(PYTEST) tests/test_temporal_tuning.py -q

test-analysis:
	$(PYTEST) tests/test_model_analysis.py -q

test-serving:
	$(PYTEST) tests/test_serving.py tests/test_api.py -q

api: deployment-config
	$(FASTAPI) dev app/main.py

api-prod: deployment-config
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

docker-build: deployment-config
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		$(IMAGE_NAME)

docker-stop:
	-docker stop $(CONTAINER_NAME)

docker-compose: deployment-config
	docker compose up --build

clean-cache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean: clean-cache
	@echo "✓ Cache files removed."

clean-models:
	rm -f models/best_model.joblib
	rm -f models/best_model_metadata.json
	rm -f models/best_tuned_candidate.joblib
	rm -f models/best_tuned_candidate_metadata.json
	rm -f models/deployment_config.json

clean-reports:
	rm -f reports/metrics/model_validation_metrics.csv
	rm -f reports/metrics/best_model_test_metrics.csv
	rm -f reports/metrics/best_model_confusion_default.csv
	rm -f reports/metrics/best_model_confusion_tuned.csv
	rm -f reports/metrics/temporal_cv_folds.csv
	rm -f reports/metrics/temporal_tuning_fold_metrics.csv
	rm -f reports/metrics/temporal_tuning_summary.csv
	rm -f reports/metrics/best_tuned_oof_predictions.parquet
	rm -f reports/metrics/tuned_operating_thresholds.csv
	rm -f reports/metrics/phase5_*
	rm -f reports/figures/12_validation_pr_curves.png
	rm -f reports/figures/13_validation_roc_curves.png
	rm -f reports/figures/14_best_model_confusion_matrix.png
	rm -f reports/figures/15_threshold_tradeoff.png
	rm -f reports/figures/16_calibration_curve.png
	rm -f reports/figures/17_native_feature_importance.png
	rm -f reports/figures/18_permutation_importance.png
	rm -f reports/figures/19_shap_importance.png

all: sync test pipeline
	@echo ""
	@echo "✓ Project build completed successfully."
