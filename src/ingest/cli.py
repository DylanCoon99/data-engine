import logging

import click
from src.config.schema import Config
from src.ingest.convert import convert
from src.ingest.split import split




'''
- Each stage is its own CLI command (python -m src.ingest, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order


├── ingest/                   # Stage 1: dataset download, BDD→YOLO conversion, split generation
│   ├── __init__.py
│   ├── convert.py            # BDD100K JSON → YOLO .txt labels
│   ├── split.py              # Generate scene-wise splits, write split manifests to Parquet
│   └── cli.py                # Click/Typer entry point: `python -m src.ingest`



Config File Structure
data:
  raw_dir: data/raw
  splits_dir: data/splits
  predictions_dir: data/predictions
  selections_dir: data/selections
  yolo_dir: data/yolo

splits:
  seed_fraction: 0.05
  unlabeled_fraction: 0.75
  val_fraction: 0.10
  test_fraction: 0.10

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

	# convert: provide label json path and output txt path
	convert(config.data)
	# split
	split(config)


	return



if __name__ == '__main__':
    main()