import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import yaml

from src.config.schema import Config
from src.train.prepare import _link_frames, _write_dataset_yaml, prepare


def _make_config(tmp_path):
	"""Create a Config with test paths and fake data on disk."""
	splits_dir = tmp_path / "splits"
	splits_dir.mkdir()
	selections_dir = tmp_path / "selections"
	selections_dir.mkdir()
	yolo_dir = tmp_path / "yolo"
	yolo_dir.mkdir()

	# fake images
	images_dir = tmp_path / "images" / "100k" / "train"
	images_dir.mkdir(parents=True)

	seed_ids = [f"seed-{i:03d}" for i in range(5)]
	mined_ids = [f"mined-{i:03d}" for i in range(3)]
	random_ids = [f"random-{i:03d}" for i in range(3)]
	val_ids = [f"val-{i:03d}" for i in range(4)]
	all_ids = seed_ids + mined_ids + random_ids + val_ids

	for fid in all_ids:
		(images_dir / f"{fid}.jpg").write_text("fake image")
		(yolo_dir / f"{fid}.txt").write_text("2 0.5 0.5 0.2 0.3\n")

	# write split manifests
	pl.DataFrame({"frame_id": seed_ids}).write_parquet(splits_dir / "seed.parquet")
	pl.DataFrame({"frame_id": val_ids}).write_parquet(splits_dir / "val.parquet")
	pl.DataFrame({"frame_id": mined_ids}).write_parquet(selections_dir / "mined.parquet")
	pl.DataFrame({"frame_id": random_ids}).write_parquet(selections_dir / "random.parquet")

	# write categories.json
	categories_path = tmp_path / "configs"
	categories_path.mkdir()
	(categories_path / "categories.json").write_text(json.dumps({
		"pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
		"bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
		"traffic light": 8, "traffic sign": 9,
		"other person": 10, "other vehicle": 11, "trailer": 12,
	}))

	yaml_path = tmp_path / "config.yaml"
	yaml_path.write_text(f"""
data:
  raw_dir: {tmp_path}
  splits_dir: {splits_dir}
  predictions_dir: {tmp_path / "predictions"}
  selections_dir: {selections_dir}
  yolo_dir: {yolo_dir}
  image_width: 1280
  image_height: 720
splits:
  seed_fraction: 0.05
  unlabeled_fraction: 0.90
  val_fraction: 0.05
model:
  weights: yolov8n.pt
  imgsz: 640
  epochs: 10
  batch: 16
  conf_threshold: 0.25
  device: cpu
mining:
  label_budget: 3
  signals:
    - uncertainty
    - rare_class
  diversity:
    max_per_scene: 5
""")

	return Config(yaml_path), seed_ids, mined_ids, random_ids, val_ids


# --- _link_frames tests ---

class TestLinkFrames:

	def test_creates_symlinks(self, tmp_path):
		"""Symlinks are created for images and labels."""
		src_images = tmp_path / "src_img"
		src_labels = tmp_path / "src_lbl"
		dst_images = tmp_path / "dst_img"
		dst_labels = tmp_path / "dst_lbl"
		src_images.mkdir()
		src_labels.mkdir()

		(src_images / "frame-001.jpg").write_text("img")
		(src_labels / "frame-001.txt").write_text("0 0.5 0.5 0.2 0.3")

		linked = _link_frames(["frame-001"], src_images, src_labels, dst_images, dst_labels)

		assert (dst_images / "frame-001.jpg").is_symlink()
		assert (dst_labels / "frame-001.txt").is_symlink()
		assert linked == 1

	def test_skips_existing(self, tmp_path):
		"""Does not overwrite existing symlinks."""
		src_images = tmp_path / "src_img"
		src_labels = tmp_path / "src_lbl"
		dst_images = tmp_path / "dst_img"
		dst_labels = tmp_path / "dst_lbl"
		src_images.mkdir()
		src_labels.mkdir()
		dst_images.mkdir(parents=True)
		dst_labels.mkdir(parents=True)

		(src_images / "frame-001.jpg").write_text("img")
		(src_labels / "frame-001.txt").write_text("label")
		(dst_images / "frame-001.jpg").write_text("existing")
		(dst_labels / "frame-001.txt").write_text("existing")

		_link_frames(["frame-001"], src_images, src_labels, dst_images, dst_labels)

		# should not be a symlink — original file preserved
		assert not (dst_images / "frame-001.jpg").is_symlink()
		assert (dst_images / "frame-001.jpg").read_text() == "existing"

	def test_missing_source_skipped(self, tmp_path):
		"""Missing source files are silently skipped."""
		src_images = tmp_path / "src_img"
		src_labels = tmp_path / "src_lbl"
		src_images.mkdir()
		src_labels.mkdir()

		linked = _link_frames(
			["nonexistent"],
			src_images, src_labels,
			tmp_path / "dst_img", tmp_path / "dst_lbl",
		)

		assert linked == 0


# --- _write_dataset_yaml tests ---

class TestWriteDatasetYaml:

	def test_writes_valid_yaml(self, tmp_path, monkeypatch):
		"""dataset.yaml is valid and has expected keys."""
		dataset_dir = tmp_path / "dataset"
		dataset_dir.mkdir()

		cat_path = tmp_path / "categories.json"
		cat_path.write_text(json.dumps({"car": 2, "pedestrian": 0}))

		yaml_path = _write_dataset_yaml(dataset_dir, categories_path=cat_path)

		assert yaml_path.exists()
		data = yaml.safe_load(yaml_path.read_text())
		assert "path" in data
		assert data["train"] == "images/train"
		assert data["val"] == "images/val"
		assert data["names"][0] == "pedestrian"
		assert data["names"][2] == "car"


# --- prepare tests ---

class TestPrepare:

	def test_creates_both_arms(self, tmp_path, monkeypatch):
		"""Prepare creates dataset dirs for both mined and random arms."""
		monkeypatch.chdir(tmp_path)
		config, seed_ids, mined_ids, random_ids, val_ids = _make_config(tmp_path)

		prepare(config)

		mined_dir = config.data.yolo_dir / "round1_mined"
		random_dir = config.data.yolo_dir / "round1_random"

		assert (mined_dir / "dataset.yaml").exists()
		assert (random_dir / "dataset.yaml").exists()
		assert (mined_dir / "images" / "train").exists()
		assert (mined_dir / "images" / "val").exists()
		assert (mined_dir / "labels" / "train").exists()
		assert (mined_dir / "labels" / "val").exists()

	def test_train_contains_seed_plus_selected(self, tmp_path, monkeypatch):
		"""Train dir contains seed + selected frames."""
		monkeypatch.chdir(tmp_path)
		config, seed_ids, mined_ids, random_ids, val_ids = _make_config(tmp_path)

		prepare(config)

		mined_train = config.data.yolo_dir / "round1_mined" / "images" / "train"
		train_files = {f.stem for f in mined_train.iterdir()}

		for fid in seed_ids:
			assert fid in train_files
		for fid in mined_ids:
			assert fid in train_files
		assert len(train_files) == len(seed_ids) + len(mined_ids)

	def test_val_contains_val_split(self, tmp_path, monkeypatch):
		"""Val dir contains the validation split frames."""
		monkeypatch.chdir(tmp_path)
		config, seed_ids, mined_ids, random_ids, val_ids = _make_config(tmp_path)

		prepare(config)

		mined_val = config.data.yolo_dir / "round1_mined" / "images" / "val"
		val_files = {f.stem for f in mined_val.iterdir()}

		for fid in val_ids:
			assert fid in val_files
		assert len(val_files) == len(val_ids)

	def test_arms_have_different_train_data(self, tmp_path, monkeypatch):
		"""Mined and random arms have different selected frames but same seed."""
		monkeypatch.chdir(tmp_path)
		config, seed_ids, mined_ids, random_ids, val_ids = _make_config(tmp_path)

		prepare(config)

		mined_train = {f.stem for f in (config.data.yolo_dir / "round1_mined" / "images" / "train").iterdir()}
		random_train = {f.stem for f in (config.data.yolo_dir / "round1_random" / "images" / "train").iterdir()}

		# seed frames should be in both
		for fid in seed_ids:
			assert fid in mined_train
			assert fid in random_train

		# selected frames should differ
		assert mined_train != random_train

	def test_dataset_yaml_names(self, tmp_path, monkeypatch):
		"""dataset.yaml has all 13 class names."""
		monkeypatch.chdir(tmp_path)
		config, *_ = _make_config(tmp_path)

		prepare(config)

		yaml_path = config.data.yolo_dir / "round1_mined" / "dataset.yaml"
		data = yaml.safe_load(yaml_path.read_text())
		assert len(data["names"]) == 13


# --- run tests ---

class TestRun:

	@patch("src.train.run._train_arm")
	def test_skips_existing_weights(self, mock_train, tmp_path, monkeypatch):
		"""Run skips training if best.pt already exists."""
		monkeypatch.chdir(tmp_path)
		config, *_ = _make_config(tmp_path)

		prepare(config)

		# create fake best.pt for both arms
		for arm in ["mined", "random"]:
			weights_dir = config.data.yolo_dir / "runs" / f"round1_{arm}" / "weights"
			weights_dir.mkdir(parents=True)
			(weights_dir / "best.pt").write_text("fake weights")

		from src.train.run import run
		run(config)

		mock_train.assert_not_called()

	@patch("src.train.run._train_arm")
	def test_trains_missing_arm(self, mock_train, tmp_path, monkeypatch):
		"""Run trains only the arm that doesn't have weights."""
		monkeypatch.chdir(tmp_path)
		config, *_ = _make_config(tmp_path)

		prepare(config)

		# only create mined weights
		weights_dir = config.data.yolo_dir / "runs" / "round1_mined" / "weights"
		weights_dir.mkdir(parents=True)
		(weights_dir / "best.pt").write_text("fake weights")

		mock_train.return_value = tmp_path / "fake_best.pt"

		from src.train.run import run
		run(config)

		# should only train random
		assert mock_train.call_count == 1
		assert mock_train.call_args[0][0] == "random"
