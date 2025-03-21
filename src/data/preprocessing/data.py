from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch

from src.data.dataset import DatasetRecord


@dataclass
class PreprocessedRecord(DatasetRecord):
    """Data class for preprocessed samples which is especially designed with Torch dataset compatibility in mind."""

    inputs: torch.Tensor
    metadata: Dict[str, Any]
    target_labels: Optional[List[str]] = None

    def __str__(self) -> str:
        return f"PreprocessedRecord(id={self.id}, targets={self.target_labels})"

    def __post_init__(self):
        if not isinstance(self.inputs, torch.Tensor):
            raise ValueError(f"Inputs must be a torch tensor, got {type(self.inputs)}")
        if self.target_labels is not None and not isinstance(self.target_labels, list):
            raise ValueError(f"Targets must be a list, got {type(self.target_labels)}")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(
                f"Metadata must be a dictionary, got {type(self.metadata)}"
            )
