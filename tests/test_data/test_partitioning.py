import pytest
import torch
from dataclasses import dataclass
from typing import List
from src.data.partitioning import (
    DatasetSplit,
    PartitioningConfig,
    load_data,
    create_splits,
    create_tensor_splits,
    save_metadata,
    leakage_sanitity_check,
    print_split_stats,
)


@dataclass
class Record:
    inputs: torch.Tensor
    target_labels: List[str]
    id: str


@pytest.fixture
def mock_sample_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(5):
        record = Record(inputs=torch.randn(10), target_labels=["label"], id=f"id_{i}")
        torch.save(record, data_dir / f"sample_{i}.pt")
    return data_dir


@pytest.fixture
def mock_split_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data = torch.randn(10, 5)
    for i, split in enumerate(["train", "val"]):
        split_data = DatasetSplit(
            data=data[i * 5 : (i + 1) * 5],
            targets=[[j] for j in range(i * 5, (i + 1) * 5)],
            record_ids=[f"id_{j}" for j in range(i * 5, (i + 1) * 5)],
        )
        torch.save(split_data.to_dict(), data_dir / f"{split}_split.pt")
    return data_dir


def test_dataset_split_initialization():
    data = torch.randn(5, 10)
    targets = [[i] for i in range(5)]
    record_ids = [f"id_{i}" for i in range(5)]
    ds = DatasetSplit(data, targets, record_ids)
    assert len(ds.data) == 5
    assert len(ds.targets) == 5
    assert len(ds.record_ids) == 5


def test_dataset_split_mismatched_lengths():
    with pytest.raises(ValueError):
        DatasetSplit(
            torch.randn(5, 10), [1, 2, 3], ["id_0", "id_1", "id_2", "id_3", "id_4"]
        )
    with pytest.raises(ValueError):
        DatasetSplit(torch.randn(5, 10), [1, 2, 3, 4, 5], ["id_0", "id_1"])


def test_partitioning_config_validation():
    config = PartitioningConfig(train_subsample_size=1.5)
    with pytest.raises(ValueError):
        config.validate()


def test_load_data_processed_splits(mock_split_files):
    data, targets, record_ids = load_data(mock_split_files)
    assert len(data) == 10
    assert len(targets) == 10
    assert len(record_ids) == 10


def test_load_data_samples(mock_sample_files):
    data, targets, record_ids = load_data(mock_sample_files)
    assert data.shape == (5, 10)
    assert len(targets) == 5
    assert record_ids == [f"id_{i}" for i in range(5)]


def test_create_splits_random():
    sample_ids = [f"id_{i}" for i in range(100)]
    config = PartitioningConfig(
        split_type="random", val_size=0.1, test_size=0.1, random_seed=42
    )
    splits = create_splits(sample_ids, config, targets=[])
    assert len(splits["test"]) == 10
    assert len(splits["val"]) == 10
    assert len(splits["train"]) == 80


def test_create_splits_stratified():
    sample_ids = [f"id_{i}" for i in range(100)]
    targets = ["class_0"] * 50 + ["class_1"] * 50
    config = PartitioningConfig(
        split_type="stratified", val_size=0.1, test_size=0.1, random_seed=42
    )
    splits = create_splits(sample_ids, config, targets)
    test_targets = [targets[sample_ids.index(id_)] for id_ in splits["test"]]
    assert abs(test_targets.count("class_0") - 5) <= 1
    assert abs(test_targets.count("class_1") - 5) <= 1


def test_create_splits_subsample():
    sample_ids = [f"id_{i}" for i in range(100)]
    config = PartitioningConfig(
        split_type="random", train_subsample_size=0.5, random_seed=42
    )
    splits = create_splits(sample_ids, config, targets=[])
    assert "train_subsample" in splits
    assert len(splits["train_subsample"]) == int(len(splits["train"]) * 0.5)


def test_leakage_sanity_check():
    splits = {
        "train": ["id1", "id2"],
        "val": ["id3"],
        "test": ["id4"],
        "train_subsample": ["id1"],
    }
    leakage_sanitity_check(splits)
    splits["val"].append("id1")
    with pytest.raises(ValueError):
        leakage_sanitity_check(splits)


def test_create_tensor_splits(tmp_path):
    data = torch.randn(5, 10)
    targets = [[i] for i in range(5)]
    record_ids = [f"id_{i}" for i in range(5)]
    splits = {"train": ["id_0", "id_1"], "val": ["id_2"], "test": ["id_3", "id_4"]}
    create_tensor_splits(splits, data, targets, record_ids, tmp_path)
    train_split = DatasetSplit.from_dict(torch.load(tmp_path / "train_split.pt"))
    assert len(train_split.data) == 2
    assert train_split.record_ids == ["id_0", "id_1"]


def test_save_metadata(tmp_path):
    splits = {"train": ["id1"], "val": ["id2"], "test": ["id3"]}
    config = PartitioningConfig(output_dir=tmp_path)
    save_metadata(splits, config, tmp_path)
    assert (tmp_path / "splits.json").exists()
    assert (tmp_path / "partitioning_cfg.json").exists()


def test_print_split_stats(capsys):
    splits = {"train": ["id_0", "id_1"], "val": ["id_2"], "test": ["id_3", "id_4"]}
    targets = ["A", "A", "B", "B", "C"]
    record_ids = ["id_0", "id_1", "id_2", "id_3", "id_4"]
    print_split_stats(splits, targets, record_ids)
    captured = capsys.readouterr()
    assert "Split 'train': 2 samples" in captured.out
    assert "A: 2 (100.0%)" in captured.out
    assert "B: 1 (50.0%)" in captured.out
    assert "C: 1 (50.0%)" in captured.out


def test_create_splits_stratified_no_targets_error():
    sample_ids = [f"id_{i}" for i in range(100)]
    config = PartitioningConfig(split_type="stratified")
    with pytest.raises(ValueError):
        create_splits(sample_ids, config, targets=[])


def test_main_success(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(10):
        record = Record(inputs=torch.randn(5), target_labels=["label"], id=f"id_{i}")
        torch.save(record, data_dir / f"sample_{i}.pt")
    monkeypatch.setattr(
        "sys.argv",
        [
            "script_name",
            "--data_dir",
            str(data_dir),
            "--split_type",
            "random",
            "--output_dir",
            str(tmp_path / "output"),
            "--val_size",
            "0.1",
            "--test_size",
            "0.1",
        ],
    )
    from src.data.partitioning import main

    main()
    output_dir = tmp_path / "output"
    assert (output_dir / "train_split.pt").exists()
    assert (output_dir / "splits.json").exists()
