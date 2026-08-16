import logging
from itertools import batched

import polars as pl
from ultralytics import YOLO

from src.config.schema import Config
from src.config.constants import TRAIN_IMAGES_DIR

logger = logging.getLogger(__name__)


def batch_inference(config: Config):

	# get model hyperparameters
	model = YOLO(config.model.weights)
	conf_threshold = config.model.conf_threshold
	batch_size = config.model.batch

	# get paths
	images_dir = config.data.raw_dir / TRAIN_IMAGES_DIR
	predictions_dir = config.data.predictions_dir
	predictions_dir.mkdir(parents=True, exist_ok=True)

	# read the unlabeled split manifest
	unlabeled = pl.read_parquet(config.data.splits_dir / "unlabeled.parquet")
	frame_ids = unlabeled["frame_id"].to_list()
	logger.info("Loaded %d unlabeled frame IDs", len(frame_ids))

	# batch the frame IDs
	batches = [list(batch) for batch in batched(frame_ids, batch_size)]
	logger.info("Split into %d batches of size %d", len(batches), batch_size)

	skipped = 0

	for i, batch_ids in enumerate(batches):
		shard_path = predictions_dir / f"shard_{i:04d}.parquet"
		if shard_path.exists():
			skipped += 1
			continue

		# resolve frame IDs to image paths
		image_paths = [str(images_dir / (fid + ".jpg")) for fid in batch_ids]

		# perform inference
		results = model.predict(image_paths, conf=conf_threshold, verbose=False)

		# collect detections into rows
		rows = []
		for j, result in enumerate(results):
			frame_id = batch_ids[j]

			if len(result.boxes) == 0:
				# keep a row so we know this frame was processed
				rows.append({
					"frame_id": frame_id,
					"class_id": -1,
					"confidence": 0.0,
					"x_center": 0.0,
					"y_center": 0.0,
					"width": 0.0,
					"height": 0.0,
				})
				continue

			for box in result.boxes:
				rows.append({
					"frame_id": frame_id,
					"class_id": int(box.cls.item()),
					"confidence": box.conf.item(),
					"x_center": box.xywhn[0][0].item(),
					"y_center": box.xywhn[0][1].item(),
					"width": box.xywhn[0][2].item(),
					"height": box.xywhn[0][3].item(),
				})

		# write shard
		df = pl.DataFrame(rows)
		df.write_parquet(shard_path)

		if (i + 1) % 50 == 0:
			logger.info("Processed %d / %d batches", i + 1, len(batches))

	if skipped > 0:
		logger.info("Skipped %d existing shards", skipped)
	logger.info("Inference complete. Shards written to %s", predictions_dir)
