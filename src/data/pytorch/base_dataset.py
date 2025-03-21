from abc import ABC
from pathlib import Path
import re
from typing import Optional, Tuple, Union, List

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from src.data.dataset import DatasetModality
from src.data.partitioning import DatasetSplit
from src.data.unified import UnifiedDataset

# Base splits and regex pattern for fold splits (e.g., fold_1_train, fold_2_val)
SUPPORTED_SPLIT_NAMES = ["train", "val", "test", "train_subsample"]
FOLD_SPLIT_PATTERN = re.compile(r"fold_\d+_(train|val)")


class BaseTorchDataset(ABC, Dataset):
    """PyTorch Dataset for preprocessed medical data using split-level tensors.

    Attributes:
        data_root: Path to the dataset root directory
        split: Dataset split(s) to use (train/val/test or fold_N_train/fold_N_val)
        apply_transforms: Whether to apply data augmentation transforms
        device: Device to load tensors to
    """

    def __init__(
        self,
        data_root: str,
        modality: DatasetModality,
        dataset_key: str,
        split: str,
        apply_transforms: bool = True,
        device: Optional[Union[str, torch.device]] = "cpu",
        use_train_subsample: bool = False,
    ) -> None:
        if not self._is_valid_split(split):
            raise ValueError(
                f"Unsupported split: {split}. Must be one of {SUPPORTED_SPLIT_NAMES} "
                "or follow the pattern 'fold_N_train' or 'fold_N_val' where N is a positive integer"
            )

        data_root = Path(data_root).absolute()
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        self.processed_dataset = UnifiedDataset(data_root, modality, dataset_key)
        if not self.processed_dataset.has_splits():
            raise ValueError(
                f"Dataset does not have splits: {data_root}. Partition the dataset first."
            )
        self.data_root = self.processed_dataset.paths["processed"]

        self.split_name = split
        if use_train_subsample and split == "train":
            self.split_name = "train_subsample"

        self.device = device
        self.apply_transforms = apply_transforms

        self._initialize_label_encoder()
        self._load_split_data()

    @staticmethod
    def _is_valid_split(split: str) -> bool:
        """Check if the split name is valid.

        Args:
            split: Split name to validate

        Returns:
            bool: True if the split name is valid
        """
        if split in SUPPORTED_SPLIT_NAMES:
            return True
        return bool(FOLD_SPLIT_PATTERN.match(split))

    def _initialize_label_encoder(self) -> None:
        """Initialize the MultiLabelBinarizer with all possible labels."""
        dataset_info = self.processed_dataset.get_dataset_info()
        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([dataset_info["target_labels"]])

    def _load_split_data(self) -> None:
        """Load and combine data from multiple splits."""
        split_file = self.data_root / f"{self.split_name}_split.pt"
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")

        data_dict = torch.load(split_file, map_location=self.device)
        split_data = DatasetSplit.from_dict(data_dict)

        self.record_ids = split_data.record_ids
        self.data = split_data.data
        self.targets = [self.encode_targets(x) for x in split_data.targets]

        assert len(self.data) == len(self.targets) == len(self.record_ids)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample and its target by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (transformed_inputs, transformed_targets)
        """
        return self.data[idx], self.targets[idx]

    def encode_targets(self, targets: List[str]) -> torch.Tensor:
        """Encode string labels into one-hot tensor.

        Args:
            targets: List of string labels

        Returns:
            One-hot encoded tensor of shape (num_classes,)
        """
        if not targets or len(targets) == 0:
            return torch.tensor([])

        one_hot = self.label_encoder.transform([targets])
        return torch.tensor(one_hot, dtype=torch.float32).squeeze(0)

    @property
    def is_training(self) -> bool:
        """Check if the dataset is in training mode."""
        return self.split_name == "train"

    @property
    def idx_to_label(self) -> dict:
        """Get a mapping of label index to label name."""
        return {idx: label for idx, label in enumerate(self.label_encoder.classes_)}

    def get_decoded_targets(self, one_hot_targets: torch.Tensor) -> List[str]:
        """Convert one-hot tensor back to string labels.

        Args:
            one_hot_targets: One-hot encoded tensor

        Returns:
            List of decoded string labels
        """
        decoded = self.label_encoder.inverse_transform(one_hot_targets.numpy())
        return list(decoded)

    def get_label_by_label_idx(self, label_idx: int) -> str:
        """Get the label string by label index."""
        # one_hot = torch.zeros(len(self.label_encoder.classes_))
        #  make shape (1, num_classes)
        one_hot = torch.zeros(1, len(self.label_encoder.classes_))
        one_hot[0, label_idx] = 1

        decoded_targets = self.get_decoded_targets(one_hot)
        assert len(decoded_targets) == 1
        # return the first element
        return decoded_targets[0][0]

    def get_raw_sample_id(self, idx: int) -> str:
        """Get the original sample ID for a given index."""
        if idx >= len(self.record_ids):
            raise ValueError(f"Index out of bounds: {idx} >= {len(self.record_ids)}")
        return self.record_ids[idx]

    def get_idx_by_record_id(self, record_id: str) -> int:
        """Get the index of a sample by its record ID."""
        if record_id not in self.record_ids:
            raise ValueError(f"Record ID not found in dataset: {record_id}")
        return self.record_ids.index(record_id)

    @property
    def y(self) -> torch.Tensor:
        """Get all targets/labels as a single tensor.

        Returns:
            torch.Tensor: A tensor of shape (n_samples, n_classes) containing all labels
        """
        return torch.stack(self.targets)
