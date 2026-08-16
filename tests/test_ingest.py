import json
import pytest
from pathlib import Path
import polars as pl

from src.config.schema import DataConfig, Config, SplitsConfig


# --- convert tests ---

class TestConvert:

	def _make_label_json(self, tmp_path):
		"""Create a minimal BDD100K label JSON and categories file."""
		labels_dir = tmp_path / "labels"
		labels_dir.mkdir()
		label_file = labels_dir / "det_v2_train_release.json"

		data = [
			{
				"videoName": "frame-001",
				"name": "frame-001.jpg",
				"labels": [
					{
						"category": "car",
						"box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
					},
					{
						"category": "pedestrian",
						"box2d": {"x1": 500, "y1": 100, "x2": 550, "y2": 250},
					},
				],
			},
			{
				"videoName": "frame-002",
				"name": "frame-002.jpg",
				"labels": [
					{
						"category": "bus",
						"box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360},
					},
				],
			},
			{
				"videoName": "frame-003",
				"name": "frame-003.jpg",
				"labels": None,
			},
			{
				"videoName": "frame-004",
				"name": "frame-004.jpg",
				"labels": [
					{
						"category": "truck",
						"box2d": None,
					},
				],
			},
		]

		label_file.write_text(json.dumps(data))
		return tmp_path

	def test_convert_basic(self, tmp_path, monkeypatch):
		"""Convert produces correct YOLO .txt files."""
		raw_dir = self._make_label_json(tmp_path)
		yolo_dir = tmp_path / "yolo"

		# point categories.json to a test version
		cat_path = tmp_path / "categories.json"
		cat_path.write_text(json.dumps({
			"pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
			"bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
			"traffic light": 8, "traffic sign": 9,
			"other person": 10, "other vehicle": 11, "trailer": 12,
		}))
		monkeypatch.chdir(tmp_path)
		# create configs dir with categories.json so convert can find it
		(tmp_path / "configs").mkdir()
		(tmp_path / "configs" / "categories.json").write_text(cat_path.read_text())

		data = DataConfig(
			raw_dir=raw_dir,
			yolo_dir=yolo_dir,
			image_width=1280,
			image_height=720,
		)

		from src.ingest.convert import convert
		convert(data)

		# frame-001 should have 2 lines
		f1 = yolo_dir / "frame-001.txt"
		assert f1.exists()
		lines = f1.read_text().strip().split("\n")
		assert len(lines) == 2
		# first line is car (class_id=2)
		assert lines[0].startswith("2 ")

		# frame-002 should have 1 line (bus = class_id 4)
		f2 = yolo_dir / "frame-002.txt"
		lines = f2.read_text().strip().split("\n")
		assert len(lines) == 1
		assert lines[0].startswith("4 ")

		# frame-003 had null labels — no file created
		f3 = yolo_dir / "frame-003.txt"
		assert not f3.exists()

		# frame-004 had null box2d — file created but empty
		f4 = yolo_dir / "frame-004.txt"
		assert f4.exists()
		assert f4.read_text().strip() == ""

	def test_convert_normalization(self, tmp_path, monkeypatch):
		"""Bounding box coordinates are correctly normalized."""
		raw_dir = self._make_label_json(tmp_path)
		yolo_dir = tmp_path / "yolo"

		(tmp_path / "configs").mkdir()
		(tmp_path / "configs" / "categories.json").write_text(json.dumps({
			"pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
			"bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
			"traffic light": 8, "traffic sign": 9,
			"other person": 10, "other vehicle": 11, "trailer": 12,
		}))
		monkeypatch.chdir(tmp_path)

		data = DataConfig(
			raw_dir=raw_dir,
			yolo_dir=yolo_dir,
			image_width=1280,
			image_height=720,
		)

		from src.ingest.convert import convert
		convert(data)

		# check frame-001 car: x1=100, y1=200, x2=300, y2=400
		# x_center = (100+300)/2/1280 = 0.15625
		# y_center = (200+400)/2/720 = 0.41667
		# width = (300-100)/1280 = 0.15625
		# height = (400-200)/720 = 0.27778
		lines = (yolo_dir / "frame-001.txt").read_text().strip().split("\n")
		parts = lines[0].split()
		assert len(parts) == 5
		x_c, y_c, w, h = [float(p) for p in parts[1:]]
		assert abs(x_c - 0.15625) < 0.001
		assert abs(y_c - 0.41667) < 0.001
		assert abs(w - 0.15625) < 0.001
		assert abs(h - 0.27778) < 0.001

		# all values should be between 0 and 1
		for line in lines:
			for val in [float(v) for v in line.split()[1:]]:
				assert 0.0 <= val <= 1.0

	def test_convert_missing_file(self, tmp_path, monkeypatch):
		"""Convert handles missing label file gracefully."""
		(tmp_path / "configs").mkdir()
		(tmp_path / "configs" / "categories.json").write_text(json.dumps({"car": 0}))
		monkeypatch.chdir(tmp_path)

		data = DataConfig(
			raw_dir=tmp_path / "nonexistent",
			yolo_dir=tmp_path / "yolo",
		)

		from src.ingest.convert import convert
		# should not raise, just log and return
		convert(data)


# --- split tests ---

class TestSplit:

	def _make_image_dirs(self, tmp_path, n_train=100, n_val=20):
		"""Create fake image directories with dummy .jpg files."""
		train_dir = tmp_path / "images" / "100k" / "train"
		val_dir = tmp_path / "images" / "100k" / "val"
		train_dir.mkdir(parents=True)
		val_dir.mkdir(parents=True)

		for i in range(n_train):
			(train_dir / f"frame-{i:04d}.jpg").touch()
		for i in range(n_val):
			(val_dir / f"val-{i:04d}.jpg").touch()

		return tmp_path

	def _make_config(self, tmp_path, raw_dir, seed=0.05, unlabeled=0.75, val=0.20):
		"""Create a Config object with test paths."""
		splits_dir = tmp_path / "splits"
		yaml_path = tmp_path / "config.yaml"
		yaml_path.write_text(f"""
data:
  raw_dir: {raw_dir}
  splits_dir: {splits_dir}
  predictions_dir: {tmp_path / "predictions"}
  selections_dir: {tmp_path / "selections"}
  yolo_dir: {tmp_path / "yolo"}
  image_width: 1280
  image_height: 720
splits:
  seed_fraction: {seed}
  unlabeled_fraction: {unlabeled}
  val_fraction: {val}
model:
  weights: yolov8n.pt
  imgsz: 640
  epochs: 50
  batch: 16
  conf_threshold: 0.25
mining:
  label_budget: 2000
  signals:
    - uncertainty
    - rare_class
  diversity:
    max_per_scene: 5
""")
		return Config(yaml_path)

	def test_split_sizes(self, tmp_path):
		"""Split produces correct number of frames per split."""
		raw_dir = self._make_image_dirs(tmp_path, n_train=100, n_val=20)
		config = self._make_config(tmp_path, raw_dir)

		from src.ingest.split import split
		split(config)

		splits_dir = tmp_path / "splits"
		seed = pl.read_parquet(splits_dir / "seed.parquet")
		unlabeled = pl.read_parquet(splits_dir / "unlabeled.parquet")
		val = pl.read_parquet(splits_dir / "val.parquet")
		test = pl.read_parquet(splits_dir / "test.parquet")

		assert len(seed) == 5        # 5% of 100
		assert len(unlabeled) == 75  # 75% of 100
		assert len(val) == 20        # remainder
		assert len(test) == 20       # all val images

	def test_split_no_overlap(self, tmp_path):
		"""No frame appears in more than one train split."""
		raw_dir = self._make_image_dirs(tmp_path, n_train=100, n_val=20)
		config = self._make_config(tmp_path, raw_dir)

		from src.ingest.split import split
		split(config)

		splits_dir = tmp_path / "splits"
		seed_ids = set(pl.read_parquet(splits_dir / "seed.parquet")["frame_id"].to_list())
		unlabeled_ids = set(pl.read_parquet(splits_dir / "unlabeled.parquet")["frame_id"].to_list())
		val_ids = set(pl.read_parquet(splits_dir / "val.parquet")["frame_id"].to_list())

		assert len(seed_ids & unlabeled_ids) == 0
		assert len(seed_ids & val_ids) == 0
		assert len(unlabeled_ids & val_ids) == 0

	def test_split_covers_all_train(self, tmp_path):
		"""Seed + unlabeled + val accounts for every train image."""
		raw_dir = self._make_image_dirs(tmp_path, n_train=100, n_val=20)
		config = self._make_config(tmp_path, raw_dir)

		from src.ingest.split import split
		split(config)

		splits_dir = tmp_path / "splits"
		seed_ids = pl.read_parquet(splits_dir / "seed.parquet")["frame_id"].to_list()
		unlabeled_ids = pl.read_parquet(splits_dir / "unlabeled.parquet")["frame_id"].to_list()
		val_ids = pl.read_parquet(splits_dir / "val.parquet")["frame_id"].to_list()

		all_ids = set(seed_ids + unlabeled_ids + val_ids)
		assert len(all_ids) == 100

	def test_split_deterministic(self, tmp_path):
		"""Running split twice produces identical results."""
		raw_dir = self._make_image_dirs(tmp_path, n_train=100, n_val=20)

		splits_dir_1 = tmp_path / "splits1"
		splits_dir_2 = tmp_path / "splits2"

		for splits_dir in [splits_dir_1, splits_dir_2]:
			config = self._make_config(tmp_path, raw_dir)
			# override splits_dir
			config.data.splits_dir = splits_dir
			from src.ingest.split import split
			split(config)

		for name in ["seed.parquet", "unlabeled.parquet", "val.parquet", "test.parquet"]:
			df1 = pl.read_parquet(splits_dir_1 / name)
			df2 = pl.read_parquet(splits_dir_2 / name)
			assert df1.equals(df2), f"{name} differs between runs"

	def test_split_bad_fractions(self, tmp_path):
		"""Split returns early when fractions don't sum to 1.0."""
		raw_dir = self._make_image_dirs(tmp_path, n_train=100, n_val=20)
		config = self._make_config(tmp_path, raw_dir, seed=0.5, unlabeled=0.5, val=0.5)

		from src.ingest.split import split
		split(config)

		# no parquet files should be written
		splits_dir = tmp_path / "splits"
		assert not (splits_dir / "seed.parquet").exists()
