from pathlib import Path
import json
import logging

from src.config.schema import DataConfig
from src.config.constants import TRAIN_LABELS_FILE, VAL_LABELS_FILE

logger = logging.getLogger(__name__)



'''
BDD100K label JSON structure (one entry per image, 70k entries in the train file):

{
  "videoName": "0000f77c-6257be58",
  "name": "0000f77c-6257be58.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {
        "x1": 49.44,
        "y1": 254.53,
        "x2": 357.81,
        "y2": 487.91
      },
      "attributes": { "occluded": false, "truncated": false }
    },
    {
      "category": "traffic light",
      "box2d": {
        "x1": 1125.90,
        "y1": 133.18,
        "x2": 1156.98,
        "y2": 210.88
      },
      "attributes": { "occluded": false, "truncated": false }
    }
  ],
  "attributes": { "weather": "clear", "timeofday": "daytime", "scene": "city street" }
}

YOLO .txt output format (one file per image, one line per box):
  class_id  x_center  y_center  width  height
All values normalized to [0, 1]. Image size is 1280x720.

Categories (10 classes):
  0: pedestrian, 1: rider, 2: car, 3: truck, 4: bus,
  5: train, 6: motorcycle, 7: bicycle, 8: traffic light, 9: traffic sign
'''




def _convert_label_file(label_path: Path, yolo_dir: Path, image_width: int, image_height: int):
	"""Convert a single BDD100K label JSON to YOLO .txt files."""
	categories = json.loads(Path("configs/categories.json").read_text())

	yolo_dir.mkdir(parents=True, exist_ok=True)

	logger.info("Loading labels from %s", label_path)

	try:
		with label_path.open("r", encoding="utf-8") as file:
			raw_labels = json.load(file)
	except FileNotFoundError:
		logger.error("Label file not found: %s", label_path)
		return

	logger.info("Loaded %d frames", len(raw_labels))

	skipped = 0
	converted = 0

	for video in raw_labels:
		videoName = video["videoName"]
		label_file = yolo_dir / (videoName + ".txt")
		labels = video["labels"]
		if labels is None:
			skipped += 1
			continue
		with label_file.open("w") as file:
			for label in labels:
				bbox = label.get("box2d")
				if bbox is None:
					continue
				class_id = categories[label["category"]]
				x1, x2, y1, y2 = bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]
				x_center = (x1 + x2) / 2 / image_width
				y_center = (y1 + y2) / 2 / image_height
				width = (x2 - x1) / image_width
				height = (y2 - y1) / image_height
				file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
		converted += 1

	logger.info("Done: %d converted, %d skipped (no labels)", converted, skipped)


def convert(data: DataConfig):
	"""Convert both train and val BDD100K labels to YOLO format."""
	w = data.image_width
	h = data.image_height

	# train labels
	_convert_label_file(data.raw_dir / TRAIN_LABELS_FILE, data.yolo_dir, w, h)

	# val labels (used as test set)
	_convert_label_file(data.raw_dir / VAL_LABELS_FILE, data.yolo_dir, w, h)