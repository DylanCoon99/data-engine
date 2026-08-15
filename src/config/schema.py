from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    splits_dir: Path = Path("data/splits")
    predictions_dir: Path = Path("data/predictions")
    selections_dir: Path = Path("data/selections")
    yolo_dir: Path = Path("data/yolo")
    image_width: int = 1280
    image_height: int = 720


@dataclass
class SplitsConfig:
    seed_fraction: float = 0.05
    unlabeled_fraction: float = 0.75
    val_fraction: float = 0.10
    test_fraction: float = 0.10


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16


@dataclass
class DiversityConfig:
    max_per_scene: int = 5


@dataclass
class MiningConfig:
    label_budget: int = 2000
    signals: list[str] = field(default_factory=lambda: ["uncertainty", "rare_class"])
    diversity: DiversityConfig = field(default_factory=DiversityConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)

    def __init__(self, yaml_path: str | Path):
        raw = yaml.safe_load(Path(yaml_path).read_text())

        data_raw = raw.get("data", {})
        path_fields = {"raw_dir", "splits_dir", "predictions_dir", "selections_dir", "yolo_dir"}
        self.data = DataConfig(**{
            k: Path(v) if k in path_fields else v
            for k, v in data_raw.items()
        })
        self.splits = SplitsConfig(**raw.get("splits", {}))
        self.model = ModelConfig(**raw.get("model", {}))

        mining_raw = raw.get("mining", {})
        diversity_raw = mining_raw.pop("diversity", {})
        self.mining = MiningConfig(
            **mining_raw,
            diversity=DiversityConfig(**diversity_raw),
        )
