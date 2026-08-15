from pathlib import Path

# BDD100K dataset structure (relative to raw_dir)
TRAIN_IMAGES_DIR = Path("images") / "100k" / "train"
VAL_IMAGES_DIR = Path("images") / "100k" / "val"
TRAIN_LABELS_FILE = Path("labels") / "det_v2_train_release.json"
VAL_LABELS_FILE = Path("labels") / "det_v2_val_release.json"
