import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from src.config.schema import Config


def _make_config(tmp_path):
	"""Create a Config from a temp yaml with test paths."""
	splits_dir = tmp_path / "splits"
	splits_dir.mkdir()
	predictions_dir = tmp_path / "predictions"
	images_dir = tmp_path / "images" / "100k" / "train"
	images_dir.mkdir(parents=True)

	# create fake images
	frame_ids = [f"frame-{i:04d}" for i in range(6)]
	for fid in frame_ids:
		(images_dir / f"{fid}.jpg").touch()

	# create unlabeled manifest
	df = pl.DataFrame({"frame_id": frame_ids})
	df.write_parquet(splits_dir / "unlabeled.parquet")

	yaml_path = tmp_path / "config.yaml"
	yaml_path.write_text(f"""
data:
  raw_dir: {tmp_path}
  splits_dir: {splits_dir}
  predictions_dir: {predictions_dir}
  selections_dir: {tmp_path / "selections"}
  yolo_dir: {tmp_path / "yolo"}
  image_width: 1280
  image_height: 720
splits:
  seed_fraction: 0.05
  unlabeled_fraction: 0.75
  val_fraction: 0.20
model:
  weights: yolov8n.pt
  imgsz: 640
  epochs: 50
  batch: 2
  conf_threshold: 0.25
mining:
  label_budget: 2000
  signals:
    - uncertainty
    - rare_class
  diversity:
    max_per_scene: 5
""")
	return Config(yaml_path), predictions_dir, frame_ids


def _make_mock_result(detections):
	"""Create a mock YOLO result with given detections.

	detections: list of (cls_id, confidence, x_c, y_c, w, h) tuples
	"""
	result = MagicMock()
	if not detections:
		result.boxes = []
		return result

	boxes = []
	for cls_id, conf, xc, yc, w, h in detections:
		box = MagicMock()
		box.cls.item.return_value = cls_id
		box.conf.item.return_value = conf
		box.xywhn = [torch.tensor([xc, yc, w, h])]
		boxes.append(box)

	result.boxes = boxes
	return result


class TestBatchInference:

	@patch("src.infer.batch.YOLO")
	def test_writes_shards(self, mock_yolo_cls, tmp_path):
		"""Inference writes Parquet shards to predictions dir."""
		config, predictions_dir, frame_ids = _make_config(tmp_path)

		mock_model = MagicMock()
		mock_yolo_cls.return_value = mock_model

		# each predict call gets batch_size=2 images, return 2 results
		mock_model.predict.return_value = [
			_make_mock_result([(2, 0.85, 0.5, 0.5, 0.2, 0.3)]),
			_make_mock_result([(0, 0.60, 0.3, 0.4, 0.1, 0.2)]),
		]

		from src.infer.batch import batch_inference
		batch_inference(config)

		# 6 frames / batch_size 2 = 3 shards
		shards = list(predictions_dir.glob("shard_*.parquet"))
		assert len(shards) == 3

	@patch("src.infer.batch.YOLO")
	def test_shard_schema(self, mock_yolo_cls, tmp_path):
		"""Shard parquet has the expected columns."""
		config, predictions_dir, frame_ids = _make_config(tmp_path)

		mock_model = MagicMock()
		mock_yolo_cls.return_value = mock_model
		mock_model.predict.return_value = [
			_make_mock_result([(2, 0.85, 0.5, 0.5, 0.2, 0.3)]),
			_make_mock_result([(0, 0.60, 0.3, 0.4, 0.1, 0.2)]),
		]

		from src.infer.batch import batch_inference
		batch_inference(config)

		df = pl.read_parquet(predictions_dir / "shard_0000.parquet")
		expected_cols = {"frame_id", "class_id", "confidence", "x_center", "y_center", "width", "height"}
		assert set(df.columns) == expected_cols

	@patch("src.infer.batch.YOLO")
	def test_empty_detections(self, mock_yolo_cls, tmp_path):
		"""Frames with no detections still get a row with class_id=-1."""
		config, predictions_dir, frame_ids = _make_config(tmp_path)

		mock_model = MagicMock()
		mock_yolo_cls.return_value = mock_model
		mock_model.predict.return_value = [
			_make_mock_result([]),  # no detections
			_make_mock_result([(2, 0.85, 0.5, 0.5, 0.2, 0.3)]),
		]

		from src.infer.batch import batch_inference
		batch_inference(config)

		df = pl.read_parquet(predictions_dir / "shard_0000.parquet")
		empty_rows = df.filter(pl.col("class_id") == -1)
		assert len(empty_rows) == 1
		assert empty_rows["frame_id"][0] == "frame-0000"

	@patch("src.infer.batch.YOLO")
	def test_resumable(self, mock_yolo_cls, tmp_path):
		"""Existing shards are skipped on re-run."""
		config, predictions_dir, frame_ids = _make_config(tmp_path)
		predictions_dir.mkdir(parents=True, exist_ok=True)

		# pre-create shard 0
		df = pl.DataFrame({
			"frame_id": ["frame-0000"],
			"class_id": [2],
			"confidence": [0.9],
			"x_center": [0.5],
			"y_center": [0.5],
			"width": [0.2],
			"height": [0.3],
		})
		df.write_parquet(predictions_dir / "shard_0000.parquet")

		mock_model = MagicMock()
		mock_yolo_cls.return_value = mock_model
		mock_model.predict.return_value = [
			_make_mock_result([(2, 0.85, 0.5, 0.5, 0.2, 0.3)]),
			_make_mock_result([(0, 0.60, 0.3, 0.4, 0.1, 0.2)]),
		]

		from src.infer.batch import batch_inference
		batch_inference(config)

		# predict should be called only 2 times (shards 1 and 2), not 3
		assert mock_model.predict.call_count == 2

	@patch("src.infer.batch.YOLO")
	def test_multiple_detections_per_frame(self, mock_yolo_cls, tmp_path):
		"""Multiple detections in one frame produce multiple rows."""
		config, predictions_dir, frame_ids = _make_config(tmp_path)

		mock_model = MagicMock()
		mock_yolo_cls.return_value = mock_model
		mock_model.predict.return_value = [
			_make_mock_result([
				(2, 0.85, 0.5, 0.5, 0.2, 0.3),
				(0, 0.60, 0.3, 0.4, 0.1, 0.2),
				(8, 0.40, 0.7, 0.2, 0.05, 0.1),
			]),
			_make_mock_result([(2, 0.90, 0.4, 0.6, 0.15, 0.25)]),
		]

		from src.infer.batch import batch_inference
		batch_inference(config)

		df = pl.read_parquet(predictions_dir / "shard_0000.parquet")
		frame_0_rows = df.filter(pl.col("frame_id") == "frame-0000")
		assert len(frame_0_rows) == 3
