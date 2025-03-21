#!/usr/bin/env python3

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import rootutils
import torch
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    KFold,
    StratifiedKFold,
)
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.preprocessing.data import PreprocessedRecord
import src.utils.utils as utils

logging.basicConfig(
    level=utils.get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Container for split data and metadata."""

    data: torch.Tensor
    targets: List[Optional[List[str]]]
    record_ids: List[str]

    def to_dict(self):
        return {
            "data": self.data,
            "targets": self.targets,
            "record_ids": self.record_ids,
        }

    @staticmethod
    def from_dict(data_dict):
        return DatasetSplit(
            data=data_dict["data"],
            targets=data_dict["targets"],
            record_ids=data_dict["record_ids"],
        )

    def __post_init__(self):
        if len(self.targets) != len(self.record_ids):
            raise ValueError("Targets and record IDs must have the same length")
        if len(self.data) != len(self.targets):
            raise ValueError("Data and targets must have the same length")


@dataclass
class PartitioningConfig:
    """Configuration for data partitioning."""

    split_type: str = "random"  # One of: random, stratified, kfold, stratified_kfold
    val_size: float = 0.1
    test_size: float = 0.1
    train_subsample_size: Optional[float] = None
    random_seed: int = 1337
    output_dir: Optional[Path] = None
    n_folds: int = 5  # Number of folds for k-fold cross validation

    def validate(self):
        if self.train_subsample_size and not 0 < self.train_subsample_size <= 1:
            raise ValueError("Train subsample size must be between 0 and 1.")
        if self.split_type not in ["random", "stratified", "kfold", "stratified_kfold"]:
            raise ValueError(f"Invalid split type: {self.split_type}")
        if self.split_type in ["kfold", "stratified_kfold"] and self.n_folds < 2:
            raise ValueError("Number of folds must be at least 2")

    def to_dict(self):
        return {
            "split_type": self.split_type,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "train_subsample_size": self.train_subsample_size,
            "random_seed": self.random_seed,
            "n_folds": self.n_folds,
        }


def load_data(data_dir: Path) -> Tuple[torch.Tensor, List, List[str]]:
    """Load data from directory."""
    logger.info(f"Loading data from {data_dir}")

    if any(f.name.endswith("_split.pt") for f in data_dir.glob("*_split.pt")):
        logger.debug("Loading processed splits")
        split_files = list(data_dir.glob("*_split.pt"))
        splits = [
            DatasetSplit.from_dict(torch.load(f))
            for f in tqdm(split_files, desc="Loading splits")
        ]

        combined_data = torch.cat([s.data for s in splits])
        combined_targets = []
        combined_records = []
        for split in splits:
            combined_targets.extend(split.targets)
            combined_records.extend(split.record_ids)

        return combined_data, combined_targets, combined_records

    sample_files = sorted(data_dir.glob("*.pt"))
    logger.debug(f"Loading {len(sample_files)} samples")
    records: List[PreprocessedRecord] = [
        torch.load(f) for f in tqdm(sample_files, desc="Loading samples")
    ]

    data = torch.stack([r.inputs for r in records])
    return data, [r.target_labels for r in records], [r.id for r in records]


def create_composite_labels(targets: List[Any]) -> List[str]:
    """Create composite labels for multi-label/multi-class targets.

    Args:
        targets: List of targets, where each target can be a single label or list of labels

    Returns:
        List of string labels, where multi-label targets are sorted and joined with commas
    """
    return [
        ",".join(sorted(label)) if isinstance(label, list) else str(label)
        for label in targets
    ]


def create_splits(
    sample_ids: List[str],
    config: PartitioningConfig,
    targets: List[Any],
) -> Dict[str, List[str]]:
    """Create dataset splits based on configuration."""
    logger.info(f"Creating {config.split_type} splits")

    if config.split_type in ["kfold", "stratified_kfold"]:
        if config.split_type == "kfold":
            kf = KFold(
                n_splits=config.n_folds, shuffle=True, random_state=config.random_seed
            )
            fold_indices = list(kf.split(sample_ids))
        else:  # stratified_kfold
            if not targets:
                raise ValueError("Targets required for stratified k-fold split")

            composite_labels = create_composite_labels(targets)
            kf = StratifiedKFold(
                n_splits=config.n_folds, shuffle=True, random_state=config.random_seed
            )
            fold_indices = list(kf.split(sample_ids, composite_labels))

        # Create splits dictionary with folds
        splits = {}
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            fold_num = fold_idx + 1
            splits[f"fold_{fold_num}_train"] = [sample_ids[i] for i in train_idx]
            splits[f"fold_{fold_num}_val"] = [sample_ids[i] for i in val_idx]

    elif config.split_type == "random":
        train_val, test = train_test_split(
            sample_ids, test_size=config.test_size, random_state=config.random_seed
        )
        train, val = train_test_split(
            train_val,
            test_size=config.val_size / (1 - config.test_size),
            random_state=config.random_seed,
        )
        splits = {"train": train, "val": val, "test": test}
    elif config.split_type == "stratified":
        if not targets:
            raise ValueError("Targets required for stratified split")

        composite_labels = create_composite_labels(targets)

        # First split: train_val vs test
        sss = StratifiedShuffleSplit(
            test_size=config.test_size, random_state=config.random_seed, n_splits=1
        )
        train_val_idx, test_idx = next(sss.split(sample_ids, composite_labels))

        # Second split: train vs val
        train_val_ids = [sample_ids[i] for i in train_val_idx]
        train_val_labels = [composite_labels[i] for i in train_val_idx]

        sss_val = StratifiedShuffleSplit(
            test_size=config.val_size / (1 - config.test_size),
            random_state=config.random_seed,
            n_splits=1,
        )
        train_idx, val_idx = next(sss_val.split(train_val_ids, train_val_labels))

        splits = {
            "train": [train_val_ids[i] for i in train_idx],
            "val": [train_val_ids[i] for i in val_idx],
            "test": [sample_ids[i] for i in test_idx],
        }
    else:
        raise ValueError(f"Invalid split type: {config.split_type}")

    if config.train_subsample_size and config.split_type not in [
        "kfold",
        "stratified_kfold",
    ]:
        logger.debug(f"Creating train subsample ({config.train_subsample_size})")
        subsample_size = int(len(splits["train"]) * config.train_subsample_size)
        splits["train_subsample"], _ = train_test_split(
            splits["train"], train_size=subsample_size, random_state=config.random_seed
        )

    return splits


def save_metadata(
    splits: Dict[str, List[str]], config: PartitioningConfig, output_dir: Path
) -> None:
    """Save split metadata as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = output_dir / "splits.json"
    with split_file.open("w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {split_file}")

    config_file = output_dir / "partitioning_cfg.json"
    with config_file.open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved partitioning config to {config_file}")


def create_tensor_splits(
    splits: Dict[str, List[str]],
    data: torch.Tensor,
    targets: List,
    record_ids: List[str],
    output_dir: Path,
) -> None:
    """Create and save tensor splits."""
    logger.info("Saving tensor splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {id_: i for i, id_ in enumerate(record_ids)}

    for split_name, split_ids in tqdm(splits.items(), desc="Processing splits"):
        missing = set(split_ids) - set(id_to_idx.keys())
        if missing:
            raise ValueError(f"Missing IDs in dataset: {missing}")

        indices = [id_to_idx[id_] for id_ in split_ids]
        split_data = DatasetSplit(
            data=data[indices],
            targets=[targets[i] for i in indices],
            record_ids=split_ids,
        )
        torch.save(split_data.to_dict(), output_dir / f"{split_name}_split.pt")

    logger.info(f"Saved tensor splits to {output_dir}")


def print_split_stats(
    splits: Dict[str, List[str]], targets: List, record_ids: List[str]
) -> None:
    """Print statistics for each split."""
    id_to_target = {id_: t for id_, t in zip(record_ids, targets)}

    for split_name, split_ids in splits.items():
        labels = []
        for id_ in split_ids:
            target = id_to_target.get(id_)
            if isinstance(target, list):
                labels.extend(target)
            else:
                labels.append(target)

        if not labels:
            print(f"Split '{split_name}': 0 samples")
            continue

        counts = Counter(labels)
        total = len(labels)
        print(f"Split '{split_name}': {len(split_ids)} samples")
        print("Label distribution:")
        max_len = max(len(str(l)) for l in counts.keys()) if counts else 0
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {str(label).ljust(max_len)}: {count} ({count/total:.1%})")
        print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["random", "stratified", "kfold", "stratified_kfold"],
        required=True,
        help="Splitting strategy",
    )
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test size")
    parser.add_argument(
        "--train_subsample_size", type=float, help="Train subsample fraction"
    )
    parser.add_argument("--random_seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    return parser.parse_args()


def leakage_sanitity_check(splits: Dict[str, List[str]]) -> None:
    """Check for leakage between splits."""
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("val", []))
    test_set = set(splits.get("test", []))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Data leakage between splits")

    if "train_subsample" in splits:
        subsample_set = set(splits["train_subsample"])
        # subsample should only contain a subset of the training set
        if not subsample_set.issubset(train_set):
            raise ValueError("Train subsample must be a subset of the training set")

        # subsample should not overlap with validation or test sets
        if subsample_set & val_set or subsample_set & test_set:
            raise ValueError("Data leakage between train subsample and other splits")


def main() -> None:
    """Main execution function."""
    args = parse_args()
    logger.info("Starting dataset partitioning")

    # interim dir
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data, targets, record_ids = load_data(data_dir)
        logger.info(f"Loaded {len(record_ids)} samples")

        config = PartitioningConfig(
            split_type=args.split_type,
            val_size=args.val_size,
            test_size=args.test_size,
            train_subsample_size=args.train_subsample_size,
            random_seed=args.random_seed,
            output_dir=output_dir,
            n_folds=args.n_folds,
        )
        splits = create_splits(record_ids, config, targets)
        leakage_sanitity_check(splits)

        create_tensor_splits(splits, data, targets, record_ids, output_dir)
        save_metadata(splits, config, output_dir)

        print_split_stats(splits, targets, record_ids)
        logger.info(f"Processing complete. Output saved to {output_dir}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
