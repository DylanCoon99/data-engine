# Directory Layout

```
data_engine/
├── Makefile                      # Orchestration: make ingest, make infer, make mine, etc.
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml                # Dependencies (ultralytics, polars, mlflow, hydra, etc.)
├── configs/
│   ├── base.yaml                 # Shared config: paths, splits, label budget, model params
│   └── round_1.yaml              # Round-specific overrides
├── src/
│   ├── ingest/                   # Stage 1: dataset download, BDD→YOLO conversion, split generation
│   │   ├── __init__.py
│   │   ├── convert.py            # BDD100K JSON → YOLO .txt labels
│   │   ├── split.py              # Generate scene-wise splits, write split manifests to Parquet
│   │   └── cli.py                # Click/Typer entry point: `python -m src.ingest`
│   ├── infer/                    # Stage 2: batch inference over unlabeled pool
│   │   ├── __init__.py
│   │   ├── batch.py              # Chunked, resumable inference → Parquet shards
│   │   └── cli.py
│   ├── mine/                     # Stage 3: signal computation + frame selection
│   │   ├── __init__.py
│   │   ├── signals.py            # Uncertainty score, rare-class score
│   │   ├── select.py             # Rank, diversity constraints, output selection manifest
│   │   └── cli.py
│   ├── train/                    # Stage 4: train both arms (mined + random)
│   │   ├── __init__.py
│   │   ├── prepare.py            # Build YOLO dataset dirs from selection manifest + seed pool
│   │   ├── run.py                # Kick off YOLOv8 training, log to MLflow
│   │   └── cli.py
│   └── evaluate/                 # Stage 5: evaluate both arms, generate report
│       ├── __init__.py
│       ├── metrics.py            # mAP comparison, per-class AP
│       ├── report.py             # Generate plots and summary
│       └── cli.py
├── scripts/
│   └── download_bdd100k.sh       # Helper to download + extract dataset
├── data/                         # .gitignored — raw + processed data lives here
│   ├── raw/                      # BDD100K as downloaded
│   │   ├── images/
│   │   └── labels/
│   ├── splits/                   # Parquet manifests: seed.parquet, unlabeled.parquet, val.parquet, test.parquet
│   ├── yolo/                     # YOLO-formatted dataset dirs (built per arm)
│   ├── predictions/              # Inference output Parquet shards
│   ├── signals/                  # Computed frame scores
│   └── selections/               # Selection manifests (DVC-versioned)
├── runs/                         # .gitignored — MLflow + YOLO training outputs
├── tests/                        # Smoke tests for each stage
│   └── test_smoke.py
└── .github/
    └── workflows/
        └── ci.yaml               # Smoke test on mini subset
```

## Data flow

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

## Splits (from 70k BDD100K train set)

| Split         | % of train | ~Frames | Purpose                                  |
|---------------|------------|---------|------------------------------------------|
| Seed pool     | 5%         | 3,500   | Initial labeled data for M0              |
| Unlabeled pool| 75%        | 52,500  | Corpus to mine from (GT withheld)        |
| Validation    | 10%        | 7,000   | Model selection during training          |
| Test          | 10%        | 7,000   | Final reported numbers (BDD100K val set) |

Note: We use BDD100K's official val (10k images) as our test set since BDD100K's
test set has no public annotations. Our validation split comes from the train set.
```
