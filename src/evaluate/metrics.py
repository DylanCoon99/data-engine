import json
import logging
import os
from pathlib import Path

import polars as pl
import yaml
from ultralytics import YOLO

from src.config.schema import Config
from src.config.constants import VAL_IMAGES_DIR

logger = logging.getLogger(__name__)


def _build_test_dataset(config: Config) -> Path:
	"""Build a YOLO dataset directory for the test set (BDD100K val images)."""

	test_dir = config.data.yolo_dir / "test_eval"
	images_dst = test_dir / "images" / "val"
	labels_dst = test_dir / "labels" / "val"
	images_dst.mkdir(parents=True, exist_ok=True)
	labels_dst.mkdir(parents=True, exist_ok=True)

	# test set = BDD100K val images
	images_src = config.data.raw_dir / VAL_IMAGES_DIR
	labels_src = config.data.yolo_dir

	# load test manifest
	test_ids = pl.read_parquet(config.data.splits_dir / "test.parquet")["frame_id"].to_list()
	logger.info("Linking %d test images", len(test_ids))

	for fid in test_ids:
		img_src = images_src / f"{fid}.jpg"
		img_dst = images_dst / f"{fid}.jpg"
		if img_src.exists() and not img_dst.exists():
			os.symlink(img_src.resolve(), img_dst)

		# test labels come from BDD100K val — we need to convert them too
		lbl_src = labels_src / f"{fid}.txt"
		lbl_dst = labels_dst / f"{fid}.txt"
		if lbl_src.exists() and not lbl_dst.exists():
			os.symlink(lbl_src.resolve(), lbl_dst)

	# write dataset.yaml
	categories = json.loads(Path("configs/categories.json").read_text())
	names = {v: k for k, v in categories.items()}

	dataset_yaml = {
		"path": str(test_dir.resolve()),
		"train": "images/val",
		"val": "images/val",
		"names": names,
	}

	yaml_path = test_dir / "dataset.yaml"
	yaml_path.write_text(yaml.dump(dataset_yaml, default_flow_style=False))

	return yaml_path


def _evaluate_arm(arm_name: str, weights_path: Path, test_yaml: Path, config: Config) -> dict:
	"""Run validation on the test set and return metrics."""

	logger.info("[%s] Evaluating %s on test set", arm_name, weights_path)

	model = YOLO(str(weights_path))

	results = model.val(
		data=str(test_yaml),
		split="val",
		device=config.model.device,
		imgsz=config.model.imgsz,
		batch=config.model.batch,
		verbose=False,
	)

	# load class names
	categories = json.loads(Path("configs/categories.json").read_text())
	id_to_name = {v: k for k, v in categories.items()}

	# per-class AP
	per_class = {}
	for i, ap in enumerate(results.box.maps):
		class_name = id_to_name.get(i, f"class_{i}")
		per_class[class_name] = float(ap)

	metrics = {
		"arm": arm_name,
		"mAP50": float(results.box.map50),
		"mAP50-95": float(results.box.map),
		"per_class_ap": per_class,
	}

	logger.info("[%s] mAP50: %.4f, mAP50-95: %.4f", arm_name, metrics["mAP50"], metrics["mAP50-95"])

	return metrics


def evaluate(config: Config) -> tuple[dict, dict]:
	"""Evaluate both arms on the test set."""

	# weights paths
	mined_weights = config.data.yolo_dir / "runs" / "round1_mined" / "weights" / "best.pt"
	random_weights = config.data.yolo_dir / "runs" / "round1_random" / "weights" / "best.pt"

	if not mined_weights.exists():
		logger.error("Mined weights not found: %s", mined_weights)
		return None, None
	if not random_weights.exists():
		logger.error("Random weights not found: %s", random_weights)
		return None, None

	# build test dataset with BDD100K val images
	test_yaml = _build_test_dataset(config)

	mined_metrics = _evaluate_arm("mined", mined_weights, test_yaml, config)
	random_metrics = _evaluate_arm("random", random_weights, test_yaml, config)

	return mined_metrics, random_metrics
