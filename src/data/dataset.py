from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Dict, Any


class DatasetModality(Enum):
    ECG = "ecg"
    CMR = "cmr"


@dataclass
class DatasetRecord(ABC):
    """Data class for processed samples."""

    id: str

    def __str__(self) -> str:
        return f"DatasetRecord(id={self.id})"


class BaseDataset(ABC):
    def __init__(self, data_root: Path):
        if not data_root.exists():
            raise ValueError(f"Data root does not exist: {data_root}")
        self.data_root = data_root

    @property
    @abstractmethod
    def dataset_key(self) -> str:
        """Unique identifier for this dataset."""
        pass

    @property
    @abstractmethod
    def modality(self) -> DatasetModality:
        """Data modality."""
        pass

    @cached_property
    def paths(self) -> Dict[str, Any]:
        paths = {
            "raw": self.data_root / "raw" / self.modality.value / self.dataset_key,
            "interim": self.data_root / "interim" / self.dataset_key,
            "processed": self.data_root / "processed" / self.dataset_key,
        }

        embeddings_base = self.data_root / "embeddings" / self.dataset_key
        embeddings_dict = {}
        if embeddings_base.exists():
            for pt_file in embeddings_base.rglob("*.pt"):
                # Use the stem (filename without extension) as the key
                key = pt_file.stem
                embeddings_dict[key] = pt_file.absolute()
        paths["embeddings"] = embeddings_dict

        # Misc paths
        paths["misc"] = {
            "splits_file": paths["processed"] / "splits.json",
            "dataset_info": paths["processed"] / "dataset_info.json",
            "partitioning_cfg": paths["processed"] / "partitioning_cfg.json",
        }

        return {k: v.absolute() if isinstance(v, Path) else v for k, v in paths.items()}

    def __str__(self) -> str:
        return f"Dataset(data_root={self.data_root}, {self.dataset_key=}, {self.modality=})"
