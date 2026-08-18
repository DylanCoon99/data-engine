import json
import logging
import os
from pathlib import Path

import polars as pl
import yaml

from src.config.schema import Config
from src.config.constants import TRAIN_IMAGES_DIR, VAL_IMAGES_DIR

logger = logging.getLogger(__name__)


def _link_frames(frame_ids: list[str], images_src: Path, labels_src: Path,
                 images_dst: Path, labels_dst: Path):
	"""Symlink images and labels for the given frame IDs into the destination dirs."""
	images_dst.mkdir(parents=True, exist_ok=True)
	labels_dst.mkdir(parents=True, exist_ok=True)

	linked = 0
	for fid in frame_ids:
		img_src = images_src / f"{fid}.jpg"
		lbl_src = labels_src / f"{fid}.txt"

		img_dst = images_dst / f"{fid}.jpg"
		lbl_dst = labels_dst / f"{fid}.txt"

		if img_src.exists() and not img_dst.exists():
			os.symlink(img_src.resolve(), img_dst)
		if lbl_src.exists() and not lbl_dst.exists():
			os.symlink(lbl_src.resolve(), lbl_dst)
			linked += 1

	return linked


def _write_dataset_yaml(dataset_dir: Path, categories_path: Path = Path("configs/categories.json")):
	"""Write the YOLO dataset.yaml config file."""
	categories = json.loads(categories_path.read_text())
	# invert to {id: name}
	names = {v: k for k, v in categories.items()}

	dataset_yaml = {
		"path": str(dataset_dir.resolve()),
		"train": "images/train",
		"val": "images/val",
		"names": names,
	}

	yaml_path = dataset_dir / "dataset.yaml"
	yaml_path.write_text(yaml.dump(dataset_yaml, default_flow_style=False))
	return yaml_path


def _prepare_arm(arm_name: str, selection_ids: list[str], seed_ids: list[str],
                 val_ids: list[str], config: Config) -> Path:
	"""Build a YOLO dataset directory for one arm (mined or random)."""

	dataset_dir = config.data.yolo_dir / f"round1_{arm_name}"
	images_src = config.data.raw_dir / TRAIN_IMAGES_DIR
	labels_src = config.data.yolo_dir
	val_images_src = config.data.raw_dir / VAL_IMAGES_DIR

	# train = seed + selected frames
	train_ids = seed_ids + selection_ids
	logger.info("[%s] Linking %d train frames (seed: %d + selected: %d)",
		arm_name, len(train_ids), len(seed_ids), len(selection_ids))

	_link_frames(
		train_ids,
		images_src, labels_src,
		dataset_dir / "images" / "train",
		dataset_dir / "labels" / "train",
	)

	# val split
	logger.info("[%s] Linking %d val frames", arm_name, len(val_ids))
	_link_frames(
		val_ids,
		images_src, labels_src,
		dataset_dir / "images" / "val",
		dataset_dir / "labels" / "val",
	)

	# write dataset.yaml
	yaml_path = _write_dataset_yaml(dataset_dir)
	logger.info("[%s] Wrote %s", arm_name, yaml_path)

	return yaml_path


def prepare(config: Config):
	"""Build YOLO dataset dirs for both arms (mined + random)."""

	# load split manifests
	splits_dir = config.data.splits_dir
	seed_ids = pl.read_parquet(splits_dir / "seed.parquet")["frame_id"].to_list()
	val_ids = pl.read_parquet(splits_dir / "val.parquet")["frame_id"].to_list()

	# load selection manifests
	selections_dir = config.data.selections_dir
	mined_ids = pl.read_parquet(selections_dir / "mined.parquet")["frame_id"].to_list()
	random_ids = pl.read_parquet(selections_dir / "random.parquet")["frame_id"].to_list()

	# prepare both arms
	mined_yaml = _prepare_arm("mined", mined_ids, seed_ids, val_ids, config)
	random_yaml = _prepare_arm("random", random_ids, seed_ids, val_ids, config)

	logger.info("Prepared both arms. YOLO configs:")
	logger.info("  Mined:  %s", mined_yaml)
	logger.info("  Random: %s", random_yaml)

	return mined_yaml, random_yaml
