from dataclasses import dataclass
from pathlib import Path

@dataclass
class Dataset:
    train_data_dir: Path = Path('/mnt/d/SlimPajama/SlimPajama28B')
    val_data_dir: Path = Path('/mnt/d/SlimPajama/SlimPajama28B')


@dataclass
class Data:
    dataset: Dataset = Dataset()
    checkpoint_dir: Path = '/mnt/d/gen_factory/llm/checkpoints'
    log_dir: str = '/mnt/d/gen_factory/llm/logs'