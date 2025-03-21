from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Generator

from numpy import ndarray

from src.data.dataset import BaseDataset, DatasetRecord


@dataclass
class RawRecord(DatasetRecord):
    """Data class for raw samples."""

    data: ndarray
    target_labels: List[str] | None
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return f"RawRecord(id={self.id}, targets={";".join(self.target_labels)})"

    def __post_init__(self):
        if not isinstance(self.data, ndarray):
            raise ValueError(f"Data must be a numpy array, got {type(self.data)}")
        if not isinstance(self.target_labels, list):
            raise ValueError(f"Targets must be a list, got {type(self.target_labels)}")
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"Metadata must be a dictionary, got {type(self.metadata)}"
            )


class RawDataset(BaseDataset, ABC):
    """Base class for handling raw medical data."""

    def __init__(self, data_root: Path):
        super().__init__(data_root)
        if not self.paths["raw"].exists():
            raise ValueError(f"Raw data path does not exist: {self.paths['raw']}")

    @abstractmethod
    def verify_data(self) -> None:
        """Verify raw data structure and contents. Throw error if invalid."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-specific metadata."""
        pass

    @abstractmethod
    def get_target_labels(self) -> List[str]:
        """Get list of target labels."""
        pass

    @abstractmethod
    def get_all_record_ids(self) -> List[str]:
        """Get all available record IDs without loading data."""
        pass

    @abstractmethod
    def load_record(self, record_id: str) -> RawRecord:
        """Load a single record."""
        pass

    @abstractmethod
    def get_stream(self) -> Generator[RawRecord, None, None]:
        """Stream raw records one at a time."""
        pass
