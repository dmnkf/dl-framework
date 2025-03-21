from typing import Dict, Type, List

from src.data.dataset import DatasetModality
from src.data.raw.data import RawDataset


class RawDatasetRegistry:
    """Registry for raw data handlers."""

    _registry: Dict[str, Dict[str, Type[RawDataset]]] = {"ecg": {}, "cmr": {}}

    @classmethod
    def register(cls, modality: str, dataset_key: str):
        """Decorator to register a raw data handler."""

        def decorator(raw_data_class: Type[RawDataset]):
            if modality not in cls._registry:
                cls._registry[modality] = {}
            cls._registry[modality][dataset_key] = raw_data_class
            return raw_data_class

        return decorator

    @classmethod
    def get_handler(cls, modality: str, dataset_key: str) -> Type[RawDataset]:
        """Get raw data handler by modality and key."""
        if modality not in cls._registry:
            raise ValueError(f"Unknown modality: {modality}")
        if dataset_key not in cls._registry[modality]:
            raise ValueError(f"Unknown dataset key for {modality}: {dataset_key}")
        return cls._registry[modality][dataset_key]

    @classmethod
    def get_modality(cls, dataset_key: str) -> DatasetModality:
        """Get modality by dataset key."""
        for modality, datasets in cls._registry.items():
            if dataset_key in datasets:
                return DatasetModality(modality)
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    @classmethod
    def list_datasets(cls) -> Dict[str, List[str]]:
        """List available datasets per modality."""
        return {
            modality: list(datasets.keys())
            for modality, datasets in cls._registry.items()
        }

    @classmethod
    def list_modalities(cls) -> List[str]:
        """List available modalities."""
        return list(cls._registry.keys())
