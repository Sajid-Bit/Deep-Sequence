from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    url_data_path: str
    file_name: str


@dataclass(frozen=True)
class DataProcessingConfig:
    file_path: Path
    window_size: int
    target: str
    index_colume: str


@dataclass(frozen=True)
class PrepareCallbackConfig:
    save_path_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


@dataclass(frozen=True)
class TrainModelConfig:
    best_model_path: Path
    update_model: Path
    params_epochs: int
    params_batch_size: int

