# Autonomy Data Engine

A pool-based active-learning data engine for autonomous driving perception. This system identifies the most informative frames within a large unlabeled driving-image corpus using ground-truth-free mining signals, selects a subset under a fixed labeling budget, and demonstrates that the selected subset yields superior model performance compared to random selection.

Mined selection achieved **+1.5% mAP50** over random selection at an identical label budget of 2,000 frames.

| Metric | Mined | Random | Delta |
|--------|-------|--------|-------|
| mAP50 | 0.2430 | 0.2282 | **+0.0148** |
| mAP50-95 | 0.1348 | 0.1263 | **+0.0084** |

The performance gains are concentrated in underrepresented object categories: motorcycle (+4.9%), rider (+2.3%), pedestrian (+2.0%), and bicycle (+1.6%). Common categories such as car, bus, and truck exhibit negligible differences between the two selection strategies, consistent with the expectation that random sampling already provides adequate coverage for well-represented classes.

---

## Overview

In large-scale autonomous driving perception systems, the volume of collected sensor data far exceeds available labeling capacity. The central challenge is not model architecture but **data curation**: determining which subset of unlabeled data, when labeled and incorporated into training, will produce the greatest improvement in model performance.

This project implements a single-round active-learning pipeline that addresses this problem. A pretrained object detector is deployed across an unlabeled corpus, and two mining signals — detection uncertainty and rare-class presence — are computed from the model's own predictions without access to ground truth. The highest-scoring frames are selected for labeling, subject to diversity constraints that prevent redundant selections from similar driving scenes.

To validate the approach, the pipeline trains two models under identical conditions: one on the mined selection and one on a random selection of equal size. Both are evaluated on a held-out test set, and the resulting mAP comparison constitutes the primary deliverable.

---

## Pipeline Architecture

The system is organized as five independent, sequentially executed stages. Each stage reads from and writes to disk via structured Parquet files, enabling independent execution and fault-tolerant resumption.

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

### Stage 1: Ingest

Converts BDD100K annotations from their native JSON format into per-image YOLO label files. Partitions the 70,000 training images into three disjoint subsets: a seed pool (5%), an unlabeled mining pool (90%), and a validation set (5%). The 10,000 BDD100K validation images serve as the held-out test set. All split manifests are persisted as Parquet files.

### Stage 2: Infer

Executes batch inference with a YOLOv8-nano model across the entire unlabeled pool. Predictions are written as Parquet shards, and the process is designed to be resumable: existing shards are detected and skipped upon restart, eliminating the need to reprocess completed batches in the event of interruption.

### Stage 3: Mine

Computes two ground-truth-free mining signals for each frame:

- **Detection uncertainty**: the count of detections whose confidence scores fall within an ambiguous band (0.3 to 0.7), indicating frames where the model exhibits genuine confusion.
- **Rare-class presence**: the sum of inverse corpus-wide class frequencies for each detection in the frame, assigning higher scores to frames containing underrepresented object categories.

Both signals are normalized to [0, 1] via percentile ranking and combined with equal weighting. A diversity constraint enforces a maximum number of selections per video to prevent redundant frame clusters. The output is a selection manifest of 2,000 frame identifiers. A random baseline manifest of equal size is generated in parallel, subject to the same diversity constraint.

### Stage 4: Train

Constructs two YOLO-format dataset directories — one for each experimental arm — by combining the seed pool with the respective selection manifest. Both models are initialized from identical pretrained weights and trained with identical hyperparameters, epochs, and random seeds. The sole experimental variable is the composition of the training subset.

### Stage 5: Evaluate

Evaluates both trained models on the held-out test set (10,000 images) and computes mAP50, mAP50-95, and per-class average precision. Generates a comparison summary, per-class bar chart, and overall mAP visualization.

---

## Per-Class Results

| Class | Mined AP | Random AP | Delta |
|-------|----------|-----------|-------|
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

---

## Reproduction

### Prerequisites

- Python 3.10 or later
- [BDD100K dataset](https://www.kaggle.com/datasets/awsaf49/bdd100k-dataset) (approximately 7 GB)

### Installation

```bash
pip install -e .
```

### Executing the Full Pipeline

```bash
make pipeline
```

Individual stages may also be executed independently:

```bash
make ingest       # Convert labels and generate split manifests
make infer        # Batch inference over the unlabeled pool
make mine         # Compute mining signals and select frames
make train        # Train both experimental arms
make evaluate     # Evaluate and compare results
```

### Running Tests

```bash
make test
```

---

## Project Structure

```
├── Makefile                      # Pipeline orchestration
├── configs/
│   ├── base.yaml                 # Pipeline configuration
│   └── categories.json           # BDD100K class identifier mapping
├── src/
│   ├── config/                   # Configuration dataclass and constants
│   ├── ingest/                   # Label conversion and split generation
│   ├── infer/                    # Resumable batch inference to Parquet
│   ├── mine/                     # Signal computation and frame selection
│   ├── train/                    # Dataset preparation and model training
│   └── evaluate/                 # Metric computation and report generation
└── tests/                        # Unit tests for each pipeline stage
```

---

## Technology Stack

| Concern | Implementation |
|---------|---------------|
| Object detection | Ultralytics YOLOv8-nano |
| Data serialization | Apache Parquet via Polars |
| Command-line interface | Click |
| Configuration management | YAML with Python dataclasses |
| Pipeline orchestration | GNU Make |
| Testing framework | pytest |

---

## Dataset

This project uses **BDD100K**, a large-scale diverse driving dataset comprising 100,000 dashcam images annotated with 2D bounding boxes across 13 object categories. The 70,000 training images are partitioned into seed, unlabeled, and validation subsets. The 10,000 official validation images are reserved as the held-out test set, as test set annotations are not publicly available.

---

## Configuration

All pipeline parameters are specified in `configs/base.yaml`:

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
