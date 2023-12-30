from typing import Optional, Tuple
from pathlib import Path
import glob
import random
from torch.utils.data import DataLoader
from .packed_dataset import PackedDataset, CombinedDataset

# Todo fix correct prefix
train_data_config = [
    ("", 1)
    # ("train_slim", 0.693584),
    # ("train_star", 0.306416),
]

val_data_config = [
    ("validation", 1.0),
]


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    
    dummy_global_rank = 0
    dummy_world_size = 1

    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+dummy_global_rank,
            num_processes=dummy_world_size,
            process_rank=dummy_global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader