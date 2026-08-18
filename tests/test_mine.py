import polars as pl
import pytest

from src.config.schema import Config


def _make_config(tmp_path, label_budget=5, max_per_scene=2):
	"""Create a Config with test paths."""
	predictions_dir = tmp_path / "predictions"
	predictions_dir.mkdir(parents=True, exist_ok=True)
	selections_dir = tmp_path / "selections"
	signals_dir = tmp_path / "signals"

	yaml_path = tmp_path / "config.yaml"
	yaml_path.write_text(f"""
data:
  raw_dir: {tmp_path}
  splits_dir: {tmp_path / "splits"}
  predictions_dir: {predictions_dir}
  selections_dir: {selections_dir}
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
  batch: 16
  conf_threshold: 0.25
mining:
  label_budget: {label_budget}
  signals:
    - uncertainty
    - rare_class
  diversity:
    max_per_scene: {max_per_scene}
""")
	return Config(yaml_path)


def _make_predictions(predictions_dir, rows):
	"""Write a single prediction shard from a list of dicts."""
	df = pl.DataFrame(rows)
	df.write_parquet(predictions_dir / "shard_0000.parquet")


# --- signals tests ---

class TestComputeScores:

	def test_output_schema(self, tmp_path):
		"""Scores dataframe has expected columns."""
		config = _make_config(tmp_path)
		_make_predictions(config.data.predictions_dir, [
			{"frame_id": "aaa-001", "class_id": 2, "confidence": 0.5, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3},
			{"frame_id": "bbb-001", "class_id": 0, "confidence": 0.9, "x_center": 0.3, "y_center": 0.4, "width": 0.1, "height": 0.2},
		])

		from src.mine.signals import compute_scores
		scores = compute_scores(config)

		expected = {"frame_id", "uncertainty", "rare_class", "uncertainty_norm", "rare_class_norm", "combined"}
		assert set(scores.columns) == expected

	def test_uncertain_frame_scores_higher(self, tmp_path):
		"""A frame with ambiguous detections should score higher on uncertainty."""
		config = _make_config(tmp_path)
		_make_predictions(config.data.predictions_dir, [
			# frame with uncertain detections (confidence in 0.3-0.7)
			{"frame_id": "aaa-001", "class_id": 2, "confidence": 0.45, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3},
			{"frame_id": "aaa-001", "class_id": 0, "confidence": 0.35, "x_center": 0.3, "y_center": 0.4, "width": 0.1, "height": 0.2},
			{"frame_id": "aaa-001", "class_id": 8, "confidence": 0.55, "x_center": 0.7, "y_center": 0.2, "width": 0.05, "height": 0.1},
			# frame with confident detections
			{"frame_id": "bbb-001", "class_id": 2, "confidence": 0.95, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3},
			{"frame_id": "bbb-001", "class_id": 2, "confidence": 0.92, "x_center": 0.3, "y_center": 0.4, "width": 0.1, "height": 0.2},
		])

		from src.mine.signals import compute_scores
		scores = compute_scores(config)

		uncertain = scores.filter(pl.col("frame_id") == "aaa-001")["uncertainty"][0]
		confident = scores.filter(pl.col("frame_id") == "bbb-001")["uncertainty"][0]
		assert uncertain > confident

	def test_rare_class_scores_higher(self, tmp_path):
		"""A frame with a rare class should score higher on rare_class signal."""
		config = _make_config(tmp_path)
		# class 5 (train) appears once, class 2 (car) appears many times
		rows = [
			{"frame_id": "aaa-001", "class_id": 5, "confidence": 0.8, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3},
		]
		# add many car detections across other frames
		for i in range(20):
			rows.append({"frame_id": f"bbb-{i:03d}", "class_id": 2, "confidence": 0.9, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3})

		_make_predictions(config.data.predictions_dir, rows)

		from src.mine.signals import compute_scores
		scores = compute_scores(config)

		rare_frame = scores.filter(pl.col("frame_id") == "aaa-001")["rare_class"][0]
		common_frame = scores.filter(pl.col("frame_id") == "bbb-000")["rare_class"][0]
		assert rare_frame > common_frame

	def test_sentinel_rows_excluded(self, tmp_path):
		"""Frames with only sentinel rows (class_id=-1) get rare_class=0."""
		config = _make_config(tmp_path)
		_make_predictions(config.data.predictions_dir, [
			{"frame_id": "aaa-001", "class_id": -1, "confidence": 0.0, "x_center": 0.0, "y_center": 0.0, "width": 0.0, "height": 0.0},
			{"frame_id": "bbb-001", "class_id": 2, "confidence": 0.5, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3},
		])

		from src.mine.signals import compute_scores
		scores = compute_scores(config)

		sentinel = scores.filter(pl.col("frame_id") == "aaa-001")["rare_class"][0]
		assert sentinel == 0

	def test_normalized_between_0_and_1(self, tmp_path):
		"""Normalized scores should be between 0 and 1."""
		config = _make_config(tmp_path)
		rows = []
		for i in range(10):
			rows.append({"frame_id": f"aaa-{i:03d}", "class_id": 2, "confidence": 0.3 + i * 0.05, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3})

		_make_predictions(config.data.predictions_dir, rows)

		from src.mine.signals import compute_scores
		scores = compute_scores(config)

		assert scores["uncertainty_norm"].min() >= 0.0
		assert scores["uncertainty_norm"].max() <= 1.0
		assert scores["rare_class_norm"].min() >= 0.0
		assert scores["rare_class_norm"].max() <= 1.0
		assert scores["combined"].min() >= 0.0
		assert scores["combined"].max() <= 1.0


# --- select tests ---

class TestSelect:

	def _write_scores(self, tmp_path, frame_scores):
		"""Write a scores.parquet file. frame_scores is a list of (frame_id, combined) tuples."""
		signals_dir = tmp_path / "signals"
		signals_dir.mkdir(parents=True, exist_ok=True)
		df = pl.DataFrame({
			"frame_id": [f[0] for f in frame_scores],
			"uncertainty": [0.0] * len(frame_scores),
			"rare_class": [0.0] * len(frame_scores),
			"uncertainty_norm": [0.0] * len(frame_scores),
			"rare_class_norm": [0.0] * len(frame_scores),
			"combined": [f[1] for f in frame_scores],
		})
		df.write_parquet(signals_dir / "scores.parquet")

	def test_selects_budget_count(self, tmp_path):
		"""Select returns exactly label_budget frames."""
		config = _make_config(tmp_path, label_budget=3, max_per_scene=10)
		scores = [(f"vid{i:02d}-frame{j:02d}", 1.0 - i * 0.1) for i in range(10) for j in range(1)]
		self._write_scores(tmp_path, scores)

		from src.mine.select import select
		select(config)

		mined = pl.read_parquet(config.data.selections_dir / "mined.parquet")
		assert len(mined) == 3

	def test_diversity_cap_enforced(self, tmp_path):
		"""No video contributes more than max_per_scene frames."""
		config = _make_config(tmp_path, label_budget=10, max_per_scene=2)
		# 5 frames from same video, all high scoring
		scores = [
			("vid01-frame01", 0.99),
			("vid01-frame02", 0.98),
			("vid01-frame03", 0.97),
			("vid01-frame04", 0.96),
			("vid01-frame05", 0.95),
			("vid02-frame01", 0.50),
			("vid02-frame02", 0.49),
			("vid02-frame03", 0.48),
			("vid03-frame01", 0.40),
			("vid03-frame02", 0.39),
			("vid04-frame01", 0.30),
			("vid04-frame02", 0.29),
		]
		self._write_scores(tmp_path, scores)

		from src.mine.select import select
		select(config)

		mined = pl.read_parquet(config.data.selections_dir / "mined.parquet")
		mined_ids = mined["frame_id"].to_list()

		# count per video
		video_counts = {}
		for fid in mined_ids:
			vid = fid.split("-")[0]
			video_counts[vid] = video_counts.get(vid, 0) + 1

		for vid, count in video_counts.items():
			assert count <= 2, f"Video {vid} has {count} frames, expected <= 2"

	def test_mined_and_random_no_overlap_required(self, tmp_path):
		"""Both manifests are written and have the correct size."""
		config = _make_config(tmp_path, label_budget=3, max_per_scene=10)
		scores = [(f"vid{i:02d}-frame01", 1.0 - i * 0.1) for i in range(10)]
		self._write_scores(tmp_path, scores)

		from src.mine.select import select
		select(config)

		mined = pl.read_parquet(config.data.selections_dir / "mined.parquet")
		random_sel = pl.read_parquet(config.data.selections_dir / "random.parquet")
		assert len(mined) == 3
		assert len(random_sel) == 3

	def test_mined_selects_highest_scores(self, tmp_path):
		"""Mined selection picks the highest-scoring frames."""
		config = _make_config(tmp_path, label_budget=3, max_per_scene=10)
		scores = [
			("vid01-frame01", 0.9),
			("vid02-frame01", 0.1),
			("vid03-frame01", 0.8),
			("vid04-frame01", 0.3),
			("vid05-frame01", 0.7),
		]
		self._write_scores(tmp_path, scores)

		from src.mine.select import select
		select(config)

		mined = pl.read_parquet(config.data.selections_dir / "mined.parquet")
		mined_ids = set(mined["frame_id"].to_list())
		assert "vid01-frame01" in mined_ids  # score 0.9
		assert "vid03-frame01" in mined_ids  # score 0.8
		assert "vid05-frame01" in mined_ids  # score 0.7

	def test_random_baseline_deterministic(self, tmp_path):
		"""Random selection is deterministic across runs."""
		scores = [(f"vid{i:02d}-frame01", 0.5) for i in range(20)]

		results = []
		for run in range(2):
			config = _make_config(tmp_path / f"run{run}", label_budget=5, max_per_scene=10)
			(tmp_path / f"run{run}" / "predictions").mkdir(parents=True, exist_ok=True)
			self._write_scores(tmp_path / f"run{run}", scores)

			from src.mine.select import select
			select(config)

			random_sel = pl.read_parquet(config.data.selections_dir / "random.parquet")
			results.append(random_sel["frame_id"].to_list())

		assert results[0] == results[1]
