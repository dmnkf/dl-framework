import logging
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple, Set

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.data.dataset import DatasetModality
from src.data.raw.data import RawDataset, RawRecord
from src.data.raw.registry import RawDatasetRegistry

logger = logging.getLogger(__name__)


@dataclass
class ArrhythmiaFileHeader:
    """Structured container for ECG Arrhythmia Dataset file header"""

    id: str = ""
    age: Optional[int] = None
    is_male: Optional[bool] = None
    snomed_ct_codes: List[str] = None
    labels_metadata: List[dict] = None
    num_leads: int = 12
    freq: int = 500
    signal_len: int = 5000
    gain_lead: np.ndarray = None
    baseline: np.ndarray = None

    def __post_init__(self):
        if self.snomed_ct_codes is None:
            self.snomed_ct_codes = []
        if self.labels_metadata is None:
            self.labels_metadata = []
        if self.gain_lead is None:
            self.gain_lead = 1000 * np.ones(12)
        if self.baseline is None:
            self.baseline = np.zeros(12)


class ArrhythmiaLabelMapper:
    """Integrates data from both mapping CSVs"""

    INT_CODE_META_KEY = "integration_code"

    def __init__(
        self,
        labeling_path: Path,  # Chapman_Ningbo_ECG_DB_Labeling_Info.csv
        snomed_mapping_path: Path,
    ):  # ConditionNames_SNOMED-CT.csv
        # load and normalize both data sources
        labeling_df = pd.read_csv(labeling_path).rename(
            columns={"Snomed Code": "Snomed_CT"}
        )
        snomed_df = pd.read_csv(snomed_mapping_path)

        # merge datasets on SNOMED CT code
        self.merged_df = pd.merge(
            labeling_df,
            snomed_df,
            on="Snomed_CT",
            how="outer",
            suffixes=("_labeling", "_snomed"),
        )

        self.metadata_map = self._build_metadata_mapping()
        self.valid_codes = self._get_valid_codes()

    def _build_metadata_mapping(self) -> Dict[str, dict]:
        """Create unified metadata dictionary from merged data"""
        return {
            str(row["Snomed_CT"]): {
                "snomed_code": str(row["Snomed_CT"]),
                "acronyms": self._unique_values(
                    row, ["Acronym Name_labeling", "Acronym Name_snomed"]
                ),
                "diagnosis_names": self._unique_values(row, ["Diagnosis", "Full Name"]),
                self.INT_CODE_META_KEY: row.get("Integration Code", "Unlabeled"),
                "integration_name": row.get("Integration Name", "Unlabeled"),
                "group": row.get("Group", "Unlabeled"),
                "comment": row.get("comment", ""),
            }
            for _, row in self.merged_df.iterrows()
        }

    def _unique_values(self, row, columns: List[str]) -> List[str]:
        """Extract non-empty unique values from multiple columns"""
        values = set()
        for col in columns:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                values.add(str(val).strip())
        return sorted(list(values))

    def _get_valid_codes(self) -> Set[str]:
        """Identify codes with complete integration mapping"""
        return set(
            self.merged_df[
                (self.merged_df["Integration Code"] != "Unlabeled")
                & (self.merged_df["Integration Code"].notna())
            ]["Snomed_CT"].astype(str)
        )

    def get_metadata(self, snomed_codes: List[str]) -> List[dict]:
        """Retrieve full metadata for SNOMED codes"""
        return [
            self.metadata_map.get(code)
            for code in snomed_codes
            if code in self.valid_codes
        ]

    def map_to_integration_codes(self, snomed_codes: List[str]) -> List[str] | None:
        """Convert SNOMED codes to integration codes with validation"""
        # if all codes are invalid, return None
        if not any(code in self.valid_codes for code in snomed_codes):
            return None
        return list(
            set(
                meta.get(self.INT_CODE_META_KEY)
                for meta in self.get_metadata(snomed_codes)
            )
        )


@RawDatasetRegistry.register(DatasetModality.ECG.value, "arrhythmia")
class Arrhythmia(RawDataset):
    """Handler for raw Arrhythmia ECG data."""

    dataset_key = "arrhythmia"
    modality = DatasetModality.ECG

    MAT_GLOB_PATTERN = "**/*.mat"
    HEA_GLOB_PATTERN = "**/*.hea"

    def __init__(self, data_root: Path):
        super().__init__(data_root)

        self.data_path = self.paths["raw"]

        self.labeling_csv = (
            self.data_path
            / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
            / "Chapman_Ningbo_ECG_DB_Labeling_Info.csv"
        )
        self.snomed_mapping_csv = (
            self.data_path
            / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
            / "ConditionNames_SNOMED-CT.csv"
        )

        self.__mat_files = {
            f.stem: f.absolute() for f in self.data_path.glob(self.MAT_GLOB_PATTERN)
        }
        self.__hea_files = {
            f.stem: f.absolute() for f in self.data_path.glob(self.HEA_GLOB_PATTERN)
        }

    @cached_property
    def label_mapper(self) -> ArrhythmiaLabelMapper:
        """Lazy load label mapper only when needed."""
        if not self.labeling_csv.exists():
            raise FileNotFoundError(f"Labeling CSV missing: {self.labeling_csv}")
        if not self.snomed_mapping_csv.exists():
            raise FileNotFoundError(
                f"SNOMED mapping CSV missing: {self.snomed_mapping_csv}"
            )

        return ArrhythmiaLabelMapper(
            labeling_path=self.labeling_csv,
            snomed_mapping_path=self.snomed_mapping_csv,
        )

    def verify_data(self) -> None:
        """Verify raw data structure and contents."""
        if not self.data_path.exists():
            raise ValueError(f"Arrhythmia data path does not exist: {self.data_path}")

        if len(self.__mat_files) == 0:
            raise ValueError(f"No .mat files found in {self.data_path}")

        if len(self.__hea_files) == 0:
            raise ValueError(f"No .hea files found in {self.data_path}")

        # each .mat file should have a corresponding .hea file
        mat_stems = set(self.__mat_files.keys())
        hea_stems = set(self.__hea_files.keys())
        if mat_stems != hea_stems:
            missing_mat = mat_stems - hea_stems
            missing_hea = hea_stems - mat_stems
            raise ValueError(
                f"Missing .hea files for {missing_mat} and .mat files for {missing_hea}"
            )

        # based on https://physionet.org/content/ecg-arrhythmia/1.0.0/
        assert (
            len(self.get_all_record_ids()) == 45152
        ), f"Expected 45152 records, got {len(self.get_all_record_ids())}"

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-specific metadata."""
        return {"num_leads": 12, "sampling_rate": 500, "default_sequence_length": 5000}

    def _get_record_paths(self, record_id: str) -> Tuple[Path, Path]:
        """Get paths to .mat and .hea files for a given record ID."""
        return self.__mat_files.get(record_id), self.__hea_files.get(record_id)

    def get_target_labels_by_record(self, record_id: str) -> List[str] | None:
        """Get the label for a given record ID."""
        header = self.get_header(record_id)
        if not header.snomed_ct_codes:
            return None
        return self.label_mapper.map_to_integration_codes(header.snomed_ct_codes)

    @cache
    def get_target_labels(self) -> List[str]:
        """Get list of target labels using streaming to reduce memory usage."""
        labels = set()
        for record_id in self.get_all_record_ids():
            target_labels = self.get_target_labels_by_record(record_id)
            if target_labels:
                labels.update(target_labels)
            else:
                logger.warning(f"Missing or invalid labels for record {record_id}")
        return sorted(list(labels))

    def get_all_record_ids(self) -> List[str]:
        """Get all record IDs without loading data."""
        return list(self.__mat_files.keys())

    def get_stream(self) -> Generator[RawRecord, None, None]:
        """Stream records one at a time to reduce memory usage."""
        for record_id in self.get_all_record_ids():
            yield self.load_record(record_id)

    @cache
    def load_record(self, record_id: str) -> RawRecord:
        """Retrieve a single record by its ID.

        Args:
            record_id: The ID of the record to retrieve.

        Returns:
            A RawRecord object if found, None otherwise.
        """
        mat_file, header_file = self._get_record_paths(record_id)
        header = self._parse_header(header_file)
        metadata = {
            "age": header.age,
            "is_male": header.is_male,
            "labels_metadata": header.labels_metadata,
            "mat_file": str(mat_file.relative_to(self.data_path)),
            "header_file": str(header_file.relative_to(self.data_path)),
        }

        signal_data = self._read_ecg_mat(mat_file)

        target_labels = self.get_target_labels_by_record(record_id)
        if not target_labels:
            raise ValueError(f"Record {record_id} has no valid target labels")

        return RawRecord(
            id=record_id,
            data=signal_data,
            target_labels=target_labels,
            metadata=metadata,
        )

    @staticmethod
    def _read_ecg_mat(path: Path) -> np.ndarray:
        """Read ECG signal data from .mat file."""
        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            mat_data = loadmat(str(path))
            if "val" not in mat_data:
                raise KeyError(f"MAT file {path} does not contain 'val' key")

            return mat_data["val"].astype(np.float32)

        except ValueError as e:
            if "read length must be non-negative" in str(e):
                raise ValueError(
                    f"MAT file {path} appears to be corrupted or empty"
                ) from e
            raise RuntimeError(f"Error reading MAT file {path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading MAT file {path}: {e}") from e

    @classmethod
    def _parse_signal_properties(cls, parts: List[str]) -> tuple:
        gain = float(parts[2].split("/")[0]) if "/" in parts[2] else float(parts[2])
        baseline = float(parts[4])
        return gain, baseline

    def _parse_header(self, path: Path) -> ArrhythmiaFileHeader:
        """Parse ECG header file content."""
        with open(path, "r") as f:
            header_lines = f.readlines()
        header = ArrhythmiaFileHeader(id=path.stem)

        for line in header_lines:
            if line.startswith("#Age:"):
                try:
                    header.age = int(line.split(": ")[1].strip())
                except ValueError:
                    header.age = None
            elif line.startswith("#Sex:"):
                sex = line.split(": ")[1].strip()
                if sex in ["Male", "Female"]:
                    header.is_male = sex == "Male"
                else:
                    header.is_male = None
            elif line.startswith("#Dx:"):
                raw_codes = line.split(": ")[1].strip().split(",")
                header.snomed_ct_codes = raw_codes
                header.labels_metadata = self.label_mapper.get_metadata(raw_codes)
        return header

    @cache
    def get_header(self, record_id: str) -> ArrhythmiaFileHeader:
        """Get header information for a given record ID."""
        _, header_file = self._get_record_paths(record_id)
        return self._parse_header(header_file)
