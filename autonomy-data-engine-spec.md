# Autonomy Data Engine — Project Spec (3-Week Build)

## Summary

**What it is:** An active-learning data engine for autonomous-driving perception. It ingests a driving dataset, runs a detector across the corpus, scores every frame for how valuable labeling it would be, selects the highest-value subset under a fixed label budget, retrains on it, and measures whether that mined subset beats random sampling.

**What it does, concretely:**

1. Ingests a driving dataset and indexes it into a queryable format (Parquet).
2. Runs batch inference with a 2D object detector over the corpus and persists every prediction — chunked, resumable, cached.
3. Computes *mining signals* per frame — detection uncertainty and rare-class presence — using only model outputs, no ground truth.
4. Ranks and selects frames under a label budget (e.g. "you may label 2,000 frames"), with diversity constraints to avoid redundant selections.
5. Simulates the labeling step by revealing ground truth only for selected frames.
6. Trains the detector on the selected subset alongside an identical random-selection baseline.
7. Evaluates both arms on a held-out test set. The gap between mined and random is the result.

**The headline result:** validation mAP at a given label budget, mined selection vs. random selection. The gap between those numbers is the entire value of the project.

---

## Why this project

"Data engine" is a real, named team at Waymo, Zoox, Aurora, Tesla, and most serious robotics companies. The bottleneck in autonomy is not model architecture — it is deciding which of the petabytes of logs are worth a human's attention.

This project demonstrates infrastructure competency, not modeling skill. The detector is a fixed, off-the-shelf component. The contribution is the pipeline: ingestion, batch inference, signal computation, selection, and experiment management — each as an independent, resumable stage with clean interfaces.

Target roles: ML infrastructure, data engine, ML platform, perception infrastructure, SRE, tooling.

---

## Scope

### In scope

- One dataset (BDD100K or nuScenes camera-only)
- One detector architecture (YOLOv8)
- Two mining signals (detection uncertainty + rare-class presence)
- One active-learning round
- One comparison against a random baseline
- Infrastructure polish: containerization, CI, config management, data versioning

### Explicitly out of scope

- 3D detection or lidar processing
- Ensemble disagreement (requires training multiple models)
- Temporal inconsistency signal (requires building a tracker)
- Multiple active-learning rounds
- Multiple random seeds for the baseline (note in README that production would)
- A web UI
- Custom training loops or novel architectures
- Distributed training

### Success criteria

The project is done when:
1. `make pipeline` runs the full loop end-to-end without manual intervention.
2. Mined selection outperforms random selection at the same label budget (or the null result is diagnosed and documented).
3. A stranger can clone the repo, run `make smoke-test`, and see the pipeline complete on a mini split in under 5 minutes.

---

## Dataset

**Primary: BDD100K (2D detection subset).** Camera-only, ~70k training images with 2D bounding box annotations across 10 object categories. Modest download size (~7 GB for images). Well-distributed rare classes (motorcycle, bicycle, rider, train) make class-based mining meaningful.

**Alternative: nuScenes (camera-only, 2D projected boxes).** Use the `v1.0-mini` split for development, camera images only. Skip all lidar/radar processing. More name recognition in the AV space but more setup overhead.

**Development approach:** Build and validate the entire pipeline on a small subset first (~500 frames). Get every stage working end-to-end before scaling up.

### Splits

Partition scene-wise (BDD100K has video-level groupings) or by video ID, never by random frame sampling — nearby frames are near-duplicates, and frame-level splitting leaks information.

- **Seed pool** (~5%): labeled from the start, used to train the initial model.
- **Unlabeled pool** (~75%): the corpus to mine from. Ground truth exists but is withheld.
- **Validation** (~10%): used for model selection during training.
- **Test** (~10%): touched only for final reported numbers.

---

## Detector

**YOLOv8 (small or medium variant) via Ultralytics.** Trains in minutes on a single GPU, inference is fast, the API is minimal, and it is not the point of the project. Freeze the architecture and hyperparameters on day one — every training run must be identical except for the training subset.

Do not write your own detector. Do not tune hyperparameters between arms. The detector is a fixed component in a controlled experiment.

---

## Mining signals

Each signal maps a frame to a scalar score. Both must be computable from model outputs alone, with no access to ground truth.

### 1. Detection uncertainty

Aggregate per-detection confidence into a frame-level score:

- **Boundary mass**: count of detections with confidence in an ambiguous band (0.3–0.7). Frames where the model is neither confident nor dismissive.
- **Margin**: gap between top-1 and top-2 class scores per detection, averaged across the frame. Small margins mean genuine class confusion.

### 2. Rare-class presence

Weight frames by the inverse corpus frequency of the classes the model predicts. Rare classes (motorcycle, bicycle, train) are undersampled by definition, and random selection will almost never surface them. Cheap and consistently effective.

### Combining signals

Normalize each signal to [0,1] via percentile rank across the corpus, then take a weighted sum. Report the weights used.

### Diversity constraint

Pure top-K selection returns redundant frames from a few hard scenes. Mitigate with:

- **Per-scene/video caps**: at most K frames from any one video sequence.
- **Temporal spacing**: minimum frame gap between selections within a sequence.

These are simple, effective, and take an hour to implement. Skip embedding-based diversity for the 3-week scope.

---

## Experiment protocol

```
Round 0:
  Train M0 on the seed pool.
  Evaluate M0 on test. Record baseline mAP.

Round 1:
  Run M0 inference over the entire unlabeled pool. Cache all outputs to Parquet.
  Compute mining signals from cached outputs.
  Rank frames; apply diversity constraints; select top-B frames.

  Arm A (mined):  reveal GT for the selected B frames.
  Arm B (random): reveal GT for B frames drawn uniformly at random,
                  with the same per-scene cap for fairness.

  Train M_A and M_B from identical initialization on
    seed pool + respective selections, using identical
    hyperparameters, epochs, and seed.

  Evaluate both on the held-out test set. Record.
```

**Controls:**

- Same initialization, seed, epoch count, and augmentation for both arms.
- Mining never touches ground truth. If a signal requires labels, it is cheating.
- Test set is opened once for reporting only. Model selection uses validation.

---

## Metrics

**Primary:** mAP on the held-out test set for mined vs. random at the same label budget.

**Secondary, worth reporting:**

- Per-class AP, especially rare classes (where mining should win hardest).
- Class distribution of selected frames vs. corpus distribution.
- Signal contribution: uncertainty-only vs. rarity-only vs. combined.

---

## Architecture

```
                     ┌──────────────────┐
                     │  Dataset (raw)   │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  Ingest / index  │  frame manifest, splits,
                     │                  │  metadata → Parquet
                     └────────┬─────────┘
                              │
              ┌───────────────▼───────────────┐
              │      Batch inference          │  chunked, resumable,
              │   (model → predictions)       │  writes Parquet shards
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │      Signal computation       │  uncertainty + rarity
              │   (predictions → scores)      │  → scored frame manifest
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   Selection + diversity       │  rank, cap, sample
              │      (scores → frame IDs)     │  → selection manifest
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   Label reveal (simulated)    │  GT join on selected IDs
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   Training + evaluation       │  both arms, tracked
              └───────────────┬───────────────┘
                              │
                     ┌────────▼─────────┐
                     │  Round report    │  metrics, plots, manifests
                     └──────────────────┘
```

Each stage is a **separate CLI command** reading and writing to disk. Stages are independently runnable and resumable. This is the infrastructure story: if batch inference crashes at frame 15,000, you restart from the last completed shard, not from scratch.

---

## Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Detection | Ultralytics YOLOv8 | Off-the-shelf, fast, minimal API |
| Data format | Parquet via Polars | All intermediate artifacts are Parquet |
| Config | Hydra or simple YAML | One config per round, composable |
| Experiment tracking | MLflow | Track metrics, params, artifacts per run |
| Data versioning | DVC | Version selection manifests and split definitions |
| Orchestration | Makefile | `make ingest`, `make infer`, `make mine`, `make train`, `make pipeline` |
| Containerization | Docker | Dockerfile + docker-compose for reproducibility |
| CI | GitHub Actions | `make smoke-test` on every push using mini split |
| CLI framework | Click or Typer | Clean, documented entry points |

**Key artifact to version:** the selection manifest — the list of frame IDs chosen, with their scores. This is the intellectual content of the project. Trained weights are regenerable; selection decisions are not.

---

## Infrastructure emphasis

These are the things that signal infra/SRE competency. Invest time here over ML sophistication:

- **Resumable batch inference**: shard-based checkpointing, skip completed shards on restart
- **Idempotent stages**: running a stage twice produces the same output
- **Structured logging**: every stage logs progress, timing, and key statistics
- **Config-driven everything**: no magic numbers in code, all parameters in config files
- **Makefile orchestration**: `make pipeline` runs everything; individual targets for each stage
- **Docker**: `docker-compose up` reproduces the full environment
- **CI smoke test**: GitHub Actions runs the pipeline on a tiny split on every push
- **DVC**: selection manifests and split definitions are versioned artifacts
- **MLflow**: every training run is tracked with params, metrics, and artifact paths
- **Clean error messages**: failures tell you what went wrong and where to look

---

## Hardware

2D detection with YOLOv8 is lightweight:

- **Any GPU with 8+ GB VRAM** is comfortable. Training takes minutes, inference is fast.
- **CPU-only**: viable for development and the mini split. Slower but functional.
- **Storage**: ~20 GB for BDD100K images + predictions + checkpoints. No storage concerns.

---

## Milestones (3 weeks)

### Week 1 — Skeleton + baseline

- Dataset downloaded and indexed into Parquet
- Scene-wise splits generated and committed (DVC)
- YOLOv8 trains on seed pool, baseline mAP on test set recorded
- Every pipeline stage exists as a CLI command (stubs are fine)
- Makefile with targets for each stage
- `make smoke-test` works on a mini subset
- Docker environment set up

**Exit criterion:** `make pipeline` completes end-to-end on the mini subset.

### Week 2 — Signals, selection, and the comparison

- Batch inference runs over full unlabeled pool, writes Parquet shards, is resumable
- Uncertainty and rare-class signals computed and sanity-checked
- Diversity constraints implemented (per-scene cap + temporal spacing)
- Selection manifests versioned with DVC
- Both arms (mined + random) trained and evaluated
- MLflow tracks all runs

**Exit criterion:** first mined-vs-random comparison exists with real numbers.

### Week 3 — Polish, CI, and write-up

- GitHub Actions CI runs smoke test on every push
- Signal ablation: uncertainty-only vs. rarity-only vs. combined
- Per-class AP breakdown
- Clean README: result at the top, architecture diagram, reproduction instructions
- Code cleanup: docstrings on public interfaces, type hints on CLI commands
- Verify Docker build works from a clean state

**Exit criterion:** a stranger can clone, read the README, and run `make smoke-test` successfully.

### If you fall behind

Cut in this order:
1. Signal ablation (nice to have)
2. Per-class breakdown (nice to have)
3. Docker (important but not blocking)
4. CI (important but not blocking)

**Never cut:** the mined-vs-random comparison and the Makefile pipeline. Without the comparison there is no result. Without the Makefile there is no infra story.

---

## Risks

**No gap between mined and random.** Most likely failure. Try a smaller label budget first — mining only helps when labels are genuinely scarce. A null result, honestly diagnosed, is still a legitimate project.

**Scope creep into modeling.** Do not improve the detector. It is a fixed component. Improving it changes both arms and produces no result.

**Scope creep into more signals.** Ensemble disagreement and temporal inconsistency are interesting but expensive. Save them for a v2 if time allows after week 3.

**YOLOv8 API changes.** Pin the Ultralytics version in requirements.txt / Docker on day one.

---

## Stretch goals (only after week 3 is done)

In priority order:

1. **Multiple random seeds** for the baseline, report mean + spread
2. **Second active-learning round** using the retrained model's predictions
3. **MC-dropout uncertainty** as a third signal (cheap approximation of ensemble disagreement)
4. **Embedding-based diversity** via FAISS clustering

---

## README framing

Open with the result:

> An active-learning data engine for autonomous driving perception. Mines a driving-image corpus for the frames most worth labeling, using detection uncertainty and rare-class scoring — no ground truth required. Reaches baseline mAP with **[N]%** of the label budget of random selection.
>
> *[budget-vs-mAP chart]*

Then: architecture diagram, reproduction instructions (`make pipeline`), signal ablation.

## Resume framing

> Built an active-learning data engine for AV perception: resumable batch inference pipeline over a driving-image corpus, ground-truth-free mining signals, diversity-constrained frame selection, and a controlled mined-vs-random experiment — all orchestrated via Makefile with DVC-versioned artifacts, MLflow tracking, Docker reproducibility, and CI smoke tests. Matched random-selection mAP at **[N]%** of the label budget.
