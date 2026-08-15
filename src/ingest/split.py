# Generate scene-wise splits, write split manifests to Parquet
import logging
import random
from pathlib import Path
import polars as pl

from src.config.schema import Config
from src.config.constants import TRAIN_IMAGES_DIR, VAL_IMAGES_DIR

logger = logging.getLogger(__name__)


def split(config: Config):

	# BDD100K train (70k) → seed (5%) + unlabeled (75%) + validation (20%)                                                                         
	# BDD100K val   (10k) → test (final reporting only)


	splits = config.splits
	seed_fraction = splits.seed_fraction
	unlabeled_fraction = splits.unlabeled_fraction
	val_fraction = splits.val_fraction

	# validate fractions
	total = seed_fraction + unlabeled_fraction + val_fraction
	if abs(total - 1.0) > 0.01:
		logger.error("Split fractions sum to %.2f, expected 1.0", total)
		return

	# directories
	train_dir = config.data.raw_dir / TRAIN_IMAGES_DIR
	val_dir = config.data.raw_dir / VAL_IMAGES_DIR

	if not train_dir.exists():
		logger.error("Train image directory not found: %s", train_dir)
		return
	if not val_dir.exists():
		logger.error("Val image directory not found: %s", val_dir)
		return

	# 1. List all filenames in 100k/train/
	logger.info("Listing images from %s", train_dir)
	train_files = sorted([f.stem for f in train_dir.glob("*.jpg")])
	if not train_files:
		logger.error("No .jpg files found in %s", train_dir)
		return
	logger.info("Found %d train images", len(train_files))

	# 2. Shuffle and divide into seed / unlabeled / validation
	random.seed(42)
	random.shuffle(train_files)

	n = len(train_files)
	seed_end = int(n * seed_fraction)
	unlabeled_end = seed_end + int(n * unlabeled_fraction)

	seed_files = train_files[:seed_end]
	unlabeled_files = train_files[seed_end:unlabeled_end]
	val_files = train_files[unlabeled_end:]

	# 3. BDD100K val filenames → test split
	test_files = sorted([f.stem for f in val_dir.glob("*.jpg")])

	logger.info("Split sizes — seed: %d, unlabeled: %d, val: %d, test: %d",
		len(seed_files), len(unlabeled_files), len(val_files), len(test_files))

	# 4. Write four Parquet manifests to data/splits/
	root_dir = config.data.splits_dir
	root_dir.mkdir(parents=True, exist_ok=True)
	# write seed.parquet
	df_seed = pl.DataFrame({"frame_id": seed_files})
	df_seed.write_parquet(root_dir / "seed.parquet")
	# write unlabeled.parquet
	df_unlabeled = pl.DataFrame({"frame_id": unlabeled_files})
	df_unlabeled.write_parquet(root_dir / "unlabeled.parquet")
	# write val.parquet
	df_val = pl.DataFrame({"frame_id": val_files})
	df_val.write_parquet(root_dir / "val.parquet")
	# write test.parquet
	df_test = pl.DataFrame({"frame_id": test_files})
	df_test.write_parquet(root_dir / "test.parquet")

	logger.info("Wrote split manifests to %s", root_dir)