import logging

import click
from src.config.schema import Config
from src.infer.batch import batch_inference




'''
- Each stage is its own CLI command (python -m src.infer, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order

├── infer/                    # Stage 2: batch inference over unlabeled pool
│   ├── __init__.py
│   ├── batch.py              # Chunked, resumable inference → Parquet shards
│   └── cli.py


data:
  raw_dir: data/raw
  image_width: 1280
  image_height: 720
  splits_dir: /Volumes/T7/data-engine-dataset/splits
  predictions_dir: /Volumes/T7/data-engine-dataset/predictions
  selections_dir: /Volumes/T7/data-engine-dataset/selections
  yolo_dir: /Volumes/T7/data-engine-dataset/yolo

splits:
  # fractions of the 70k BDD100K train set
  # test split is BDD100K val (10k) — not configurable
  seed_fraction: 0.05
  unlabeled_fraction: 0.75
  val_fraction: 0.20

model:
  weights: yolov8n.pt
  imgsz: 640
  epochs: 50
  batch: 16

mining:
  label_budget: 2000
  signals:
    - uncertainty
    - rare_class
  diversity:
    max_per_scene: 5





'''

@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):

	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

	config = Config(config)

	batch_inference(config)



if __name__ == '__main__':
    main()