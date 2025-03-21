import json
from dataclasses import dataclass
from functools import cached_property, cache, lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

from src.data.dataset import DatasetModality, BaseDataset, DatasetRecord
from src.data.metadata_store import MetadataStore
from src.data.preprocessing.data import PreprocessedRecord
from src.data.raw.data import RawDataset, RawRecord
from src.data.raw.registry import RawDatasetRegistry


@dataclass
class UnifiedRecord(DatasetRecord):
    raw_record: RawRecord = None
    preprocessed_record: PreprocessedRecord = None
    embeddings: Optional[torch.Tensor] = None

    def __str__(self) -> str:
        return f"UnifiedPreprocessedRecord(id={self.id}, has_embeddings={self.embeddings is not None})"


class UnifiedDataset(BaseDataset):
    def __init__(self, data_root: Path, modality: DatasetModality, dataset_key: str):
        super().__init__(data_root)
        self._dataset_key = dataset_key
        self._modality = modality

    @property
    def dataset_key(self) -> str:
        return self._dataset_key

    @property
    def modality(self) -> DatasetModality:
        return self._modality

    @cached_property
    def metadata_store(self) -> MetadataStore:
        return MetadataStore(data_root=self.paths["processed"])

    @cached_property
    def raw_dataset(self) -> RawDataset:
        return RawDatasetRegistry.get_handler(self.modality.value, self.dataset_key)(
            self.data_root
        )

    def has_dataset_info(self) -> bool:
        return self.paths["misc"]["dataset_info"].exists()

    def get_dataset_info(self) -> Dict[str, Any]:
        if not self.has_dataset_info():
            raise ValueError("No dataset info found for this dataset.")

        with open(self.paths["misc"]["dataset_info"], "r") as f:
            return json.load(f)

    def has_splits(self) -> bool:
        return self.paths["misc"]["splits_file"].exists()

    def get_splits(self) -> Dict[str, Dict[str, Any]]:
        if not self.has_splits():
            raise ValueError("No splits found for this dataset.")

        with open(self.paths["misc"]["splits_file"], "r") as f:
            splits = json.load(f)
        return splits

    def get_split_by_record_id(self, record_id: str) -> str:
        for split_name, record_ids in self.get_splits().items():
            if record_id in record_ids:
                return split_name
        raise ValueError(f"Record ID '{record_id}' not found in any split.")

    def __get_all_record_ids_from_splits(self) -> List[str]:
        return [
            record_id for split in self.get_splits().values() for record_id in split
        ]

    def __get_all_record_ids_from_interim(self) -> List[str]:
        interim_files = self.paths["interim"].glob("*.pt")
        return [f.stem for f in interim_files]

    @cache
    def get_all_record_ids(self) -> List[str]:
        # if self.has_splits():
        #    return self.__get_all_record_ids_from_splits()
        return (
            self.__get_all_record_ids_from_interim()
        )  # interim is the ground truth for record ids

    @cache
    def __load_embeddings(self, embeddings_type: str) -> Dict[str, torch.Tensor]:
        if embeddings_type not in self.paths["embeddings"].keys():
            raise ValueError(f"Embeddings type '{embeddings_type}' not found.")

        embeddings_path = self.paths["embeddings"][embeddings_type]
        if not embeddings_path.exists():
            return {}

        return torch.load(embeddings_path, map_location="cpu")  # type: ignore

    @lru_cache(maxsize=1000)
    def __load_preprocessed_record(self, record_id: str) -> PreprocessedRecord:
        if record_id not in self.get_all_record_ids():
            raise ValueError(f"Record ID '{record_id}' not found in dataset.")

        interim_file = self.paths["interim"] / f"{record_id}.pt"
        if not interim_file.exists():
            raise ValueError(f"Interim file not found for record ID '{record_id}'.")
        return torch.load(interim_file, map_location="cpu")  # type: ignore

    def get_embeddings(
        self, record_id: str, embeddings_type: str = None
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        if embeddings_type is None:
            return {
                k: self.__load_embeddings(k).get(record_id, None)
                for k in self.paths["embeddings"].keys()
            }

        return self.__load_embeddings(embeddings_type).get(record_id, None)

    def available_metadata_fields(self) -> set[str]:
        return self.metadata_store.available_fields()

    @lru_cache(maxsize=1000)
    def __getitem__(self, record_id: str) -> UnifiedRecord:
        preprocessed_record = self.__load_preprocessed_record(record_id)

        embeddings = self.get_embeddings(record_id)
        raw_record = self.raw_dataset.load_record(record_id)

        return UnifiedRecord(
            id=record_id,
            raw_record=raw_record,
            preprocessed_record=preprocessed_record,
            embeddings=embeddings,
        )

    def verify_integrity(self) -> None:
        """Validate dataset integrity using interim files as ground truth.

        Performs the following assertions:
            1. At least one interim file exists (raises ValueError if empty)
            2. If splits exist:
                a. All split IDs must exist in interim files (raises on missing IDs)
                b. All interim IDs must exist in splits (raises on coverage mismatch)
                c. No duplicate IDs across splits (raises on duplicates)
                d. Split data must be stored as lists (raises on format error)
            3. Embeddings files (if exist):
                a. Must not contain IDs absent from interim files (raises on extra IDs)
            4. Metadata:
                a. Must exist for every interim ID (raises on missing metadata)

        Raises:
            ValueError: For any integrity check failure, with detailed message
            FileNotFoundError: If critical path components are missing

        Note:
            Interim files (*.pt in interim directory) are considered the canonical
            source of truth for valid record IDs. All other components (splits,
            embeddings, metadata) must align with these IDs.
        """
        interim_ids = set(self.get_all_record_ids())

        # First check there's at least one interim record
        if not interim_ids:
            raise ValueError(f"No interim files found in {self.paths['interim']}")

        # Check if every raw record was preprocessed
        raw_record_ids = set(self.raw_dataset.get_all_record_ids())
        missing_preprocessed = raw_record_ids - interim_ids
        if missing_preprocessed:
            raise ValueError(
                f"{len(missing_preprocessed)} raw records missing from interim files. "
                f"First 5: {sorted(missing_preprocessed)[:5]}"
            )

        # Split consistency checks (if splits exist)
        if self.has_splits():
            splits = self.get_splits()
            split_ids = set()
            all_split_ids = []

            # Collect all split IDs and check per-split validity
            for split_name, split_records in splits.items():
                if not isinstance(split_records, list):
                    raise ValueError(
                        f"Split '{split_name}' should be a list of IDs, got {type(split_records)}"
                    )

                current_split = set(split_records)
                split_ids.update(current_split)
                all_split_ids.extend(split_records)

                # Check individual split validity
                missing_in_interim = current_split - interim_ids
                if missing_in_interim:
                    raise ValueError(
                        f"Split '{split_name}' contains {len(missing_in_interim)} "
                        f"IDs missing from interim files. First 5: {sorted(missing_in_interim)[:5]}"
                    )

            # Check full split coverage
            if split_ids != interim_ids:
                missing_in_splits = interim_ids - split_ids
                extra_in_splits = split_ids - interim_ids
                error_msg = []
                if missing_in_splits:
                    error_msg.append(
                        f"{len(missing_in_splits)} interim IDs missing from splits"
                    )
                if extra_in_splits:
                    error_msg.append(
                        f"{len(extra_in_splits)} extra IDs in splits not in interim"
                    )
                raise ValueError("Split coverage mismatch: " + ", ".join(error_msg))

            # Check for duplicate IDs across splits
            duplicate_ids = {id for id in all_split_ids if all_split_ids.count(id) > 1}
            if duplicate_ids:
                raise ValueError(
                    f"{len(duplicate_ids)} duplicate IDs found across splits. "
                    f"First 5: {sorted(duplicate_ids)[:5]}"
                )

        # Embeddings consistency checks
        for emb_type, emb_path in self.paths["embeddings"].items():
            if emb_path.exists():
                embeddings = self.__load_embeddings(emb_type)
                if not embeddings:
                    continue  # Skip empty embeddings files

                emb_ids = set(embeddings.keys())
                extra_emb_ids = emb_ids - interim_ids
                if extra_emb_ids:
                    raise ValueError(
                        f"{emb_type} embeddings contain {len(extra_emb_ids)} IDs "
                        f"not in interim files. First 5: {sorted(extra_emb_ids)[:5]}"
                    )

        # Metadata completeness check
        missing_metadata = []
        for record_id in interim_ids:
            if not self.metadata_store.get(record_id):
                missing_metadata.append(record_id)
            if len(missing_metadata) >= 5:  # Early exit for large datasets
                break
        if missing_metadata:
            raise ValueError(
                f"Missing metadata for {len(missing_metadata)} records. "
                f"First 5: {missing_metadata[:5]}"
            )

    def __str__(self) -> str:
        return f"ProcessedDataset(data_root={self.data_root}, modality={self.modality}, dataset_key={self.dataset_key})"
