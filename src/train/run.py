import logging
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.callbacks import mlflow as mlflow_cb

# remove MLflow callbacks so Ultralytics doesn't try to log there
mlflow_cb.callbacks = {}

from src.config.schema import Config

logger = logging.getLogger(__name__)


def _train_arm(arm_name: str, dataset_yaml: Path, config: Config) -> Path:
	"""Train a YOLOv8 model for one arm."""

	logger.info("[%s] Starting training", arm_name)

	model = YOLO(config.model.weights)

	results = model.train(
		data=str(dataset_yaml),
		epochs=config.model.epochs,
		imgsz=config.model.imgsz,
		batch=config.model.batch,
		device=config.model.device,
		project=str(config.data.yolo_dir / "runs"),
		name=f"round1_{arm_name}",
		seed=42,
		verbose=True,
	)

	best_weights = Path(results.save_dir) / "weights" / "best.pt"
	logger.info("[%s] Training complete. Best weights: %s", arm_name, best_weights)

	return best_weights


def run(config: Config):
	"""Train both arms (mined + random) with identical settings."""

	mined_yaml = config.data.yolo_dir / "round1_mined" / "dataset.yaml"
	random_yaml = config.data.yolo_dir / "round1_random" / "dataset.yaml"

	if not mined_yaml.exists():
		logger.error("Mined dataset.yaml not found: %s. Run prepare first.", mined_yaml)
		return
	if not random_yaml.exists():
		logger.error("Random dataset.yaml not found: %s. Run prepare first.", random_yaml)
		return

	mined_weights = config.data.yolo_dir / "runs" / "round1_mined" / "weights" / "best.pt"
	if mined_weights.exists():
		logger.info("Mined arm already trained, skipping. Weights: %s", mined_weights)
	else:
		mined_weights = _train_arm("mined", mined_yaml, config)

	random_weights = config.data.yolo_dir / "runs" / "round1_random" / "weights" / "best.pt"
	if random_weights.exists():
		logger.info("Random arm already trained, skipping. Weights: %s", random_weights)
	else:
		random_weights = _train_arm("random", random_yaml, config)

	logger.info("Both arms trained.")
	logger.info("  Mined weights:  %s", mined_weights)
	logger.info("  Random weights: %s", random_weights)

	return mined_weights, random_weights
