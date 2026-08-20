# Autonomy Data Engine

An active-learning data engine for autonomous driving perception. Mines a 70k-frame driving-image corpus for the frames most worth labeling, using detection uncertainty and rare-class scoring — no ground truth required. Reached **+1.5% mAP50** over random selection at the same label budget.

| Metric | Mined | Random | Delta |
|--------|-------|--------|-------|
| mAP50 | 0.2430 | 0.2282 | **+0.0148** |
| mAP50-95 | 0.1348 | 0.1263 | **+0.0084** |

Mining wins hardest on rare classes — motorcycle (+4.9%), rider (+2.3%), pedestrian (+2.0%), bicycle (+1.6%) — while common classes (car, bus, truck) remain roughly equal.

## How It Works

The pipeline answers: **which frames are worth paying a human to label?**

1. **Ingest** — Converts BDD100K labels to YOLO format, splits 70k images into seed (5%), unlabeled pool (90%), and validation (5%). BDD100K val (10k) serves as the held-out test set.
2. **Infer** — Runs YOLOv8-nano over the unlabeled pool in resumable batches, caching all predictions to Parquet shards.
3. **Mine** — Computes two ground-truth-free signals per frame:
   - *Uncertainty*: count of detections with confidence in the ambiguous band (0.3–0.7)
   - *Rare-class presence*: sum of inverse class frequency for each detection

   Normalizes, combines, and selects the top 2,000 frames with per-video diversity caps.
4. **Train** — Trains two identical YOLOv8-nano models: one on seed + mined frames, one on seed + random frames. Same weights, same seed, same hyperparameters.
5. **Evaluate** — Runs both models on the held-out test set and compares mAP.

## Architecture

```
raw BDD100K → ingest → splits (Parquet) + YOLO labels
                              ↓
                seed pool → train round-0 model (M0)
                              ↓
                unlabeled pool → infer (M0) → predictions (Parquet)
                              ↓
                predictions → mine → scores → selection manifest
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
        mined frames                    random frames
        (reveal GT)                     (reveal GT)
              ↓                               ↓
        train M_mined                   train M_random
              ↓                               ↓
        eval on test                    eval on test
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                    comparison report
```

Each stage is an independent CLI command reading and writing to disk. Stages are resumable — if batch inference crashes at shard 30, restart and it skips shards 0–29.

## Per-Class Results

| Class | Mined | Random | Delta |
|-------|-------|--------|-------|
| motorcycle | 0.0867 | 0.0376 | **+0.0491** |
| rider | 0.1261 | 0.1029 | **+0.0233** |
| pedestrian | 0.2107 | 0.1912 | **+0.0195** |
| bicycle | 0.1141 | 0.0984 | **+0.0157** |
| traffic light | 0.1096 | 0.1023 | **+0.0073** |
| other vehicle | 0.0110 | 0.0049 | **+0.0061** |
| car | 0.3919 | 0.3983 | -0.0065 |
| traffic sign | 0.1790 | 0.1830 | -0.0039 |
| bus | 0.2628 | 0.2630 | -0.0002 |
| truck | 0.2603 | 0.2609 | -0.0006 |

## Quick Start

### Prerequisites

- Python 3.10+
- [BDD100K dataset](https://www.kaggle.com/datasets/awsaf49/bdd100k-dataset) (~7 GB)

### Setup

```bash
pip install -e .
```

### Run the full pipeline

```bash
make pipeline
```

Or run stages individually:

```bash
make ingest     # convert labels + generate splits
make infer      # batch inference over unlabeled pool
make mine       # compute signals + select frames
make train      # train both arms
make evaluate   # compare mined vs random
```

### Run tests

```bash
make test
```

## Project Structure

```
├── Makefile                  # Pipeline orchestration
├── configs/
│   ├── base.yaml             # All pipeline configuration
│   └── categories.json       # BDD100K class ID mapping
├── src/
│   ├── config/               # Config dataclass + constants
│   ├── ingest/               # BDD100K → YOLO conversion + splits
│   ├── infer/                # Resumable batch inference → Parquet
│   ├── mine/                 # Uncertainty + rare-class signals → selection
│   ├── train/                # Prepare YOLO datasets + train both arms
│   └── evaluate/             # mAP comparison + report generation
└── tests/                    # Unit tests for each stage
```

## Tech Stack

| Concern | Choice |
|---------|--------|
| Detection | Ultralytics YOLOv8-nano |
| Data format | Parquet via Polars |
| CLI | Click |
| Config | YAML + Python dataclasses |
| Orchestration | Makefile |
| Testing | pytest |

## Dataset

**BDD100K** — 100k dashcam images across 10 object categories. 70k train images split into seed/unlabeled/validation pools, 10k val images used as the held-out test set.

## Configuration

All parameters live in `configs/base.yaml`:

```yaml
splits:
  seed_fraction: 0.05
  unlabeled_fraction: 0.90
  val_fraction: 0.05

model:
  weights: yolov8n.pt
  epochs: 10
  batch: 16
  device: mps

mining:
  label_budget: 2000
  signals: [uncertainty, rare_class]
  diversity:
    max_per_scene: 5
```
