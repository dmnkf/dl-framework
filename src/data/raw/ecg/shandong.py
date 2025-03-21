import logging
from pathlib import Path
from typing import Dict, Any, List, Generator
from functools import cache

import numpy as np
import pandas as pd
import h5py

from src.data.dataset import DatasetModality
from src.data.raw.data import RawDataset, RawRecord
from src.data.raw.registry import RawDatasetRegistry

logger = logging.getLogger(__name__)


@RawDatasetRegistry.register(DatasetModality.ECG.value, "shandong")
class Shandong(RawDataset):
    """Handler for raw Shandong ECG data.

    Expected data structure in data_root:
        records/           # contains HDF5 files e.g. A00001.h5, A00002.h5, etc.
        metadata.csv       # CSV file with metadata (ECG_ID, AHA_Code, Patient_ID, Age, Sex, N, Date)
        code.csv           # CSV file mapping diagnostic codes to descriptions
    """

    dataset_key = "shandong"
    modality = DatasetModality.ECG

    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Path to the dataset root folder.
        """
        super().__init__(data_root)
        self.root = self.paths["raw"]

        self.records_path = self.root / "records"
        self.metadata_csv = self.root / "metadata.csv"
        self.code_csv = self.root / "code.csv"

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata and (optionally) diagnostic code mapping from CSV files."""
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")
        self.metadata_df = pd.read_csv(self.metadata_csv)
        if self.metadata_df.empty:
            raise ValueError(f"Metadata CSV is empty: {self.metadata_csv}")

        # patient_ID in metadata is not unique. Group by patient_id and use random record
        self.metadata_df = self.metadata_df.groupby("Patient_ID").sample(
            n=1, random_state=1337
        )

        if self.code_csv.exists():
            self.code_df = pd.read_csv(self.code_csv)
        else:
            self.code_df = None

        # Build a dictionary for quick lookup: mapping ECG_ID -> metadata row.
        self.metadata_dict = {
            str(row["ECG_ID"]): row for _, row in self.metadata_df.iterrows()
        }

    def verify_data(self) -> None:
        """Verify raw data structure and contents."""
        if not self.records_path.exists():
            raise ValueError(f"Records directory does not exist: {self.records_path}")

        if not self.metadata_csv.exists():
            raise ValueError(f"Metadata CSV not found: {self.metadata_csv}")

        for record_id in self.get_all_record_ids():
            record_file = self.records_path / f"{record_id}.h5"
            if not record_file.exists():
                raise FileNotFoundError(
                    f"Record {record_id} not found in records directory"
                )

        # based on https://www.nature.com/articles/s41597-022-01403-5#ref-CR15, we want only unique patients with one record per patient
        assert (
            len(self.metadata_df) == 24666
        ), f"Expected 24666 records, got {len(self.metadata_df)}"

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-specific metadata.

        Returns:
            A dictionary containing basic information about the dataset.
        """
        return {
            "num_leads": 12,
            "sampling_rate": 500,
            "record_duration_range_seconds": [10, 60],
            "total_records": len(self.metadata_df),
        }

    @cache
    def get_target_labels(self) -> List[str]:
        """Collect the unique diagnostic codes across all records.

        Returns:
            Sorted list of unique target labels.
        """
        labels = set()
        for _, row in self.metadata_df.iterrows():
            aha_codes = self._extract_aha_codes(str(row["AHA_Code"]))
            primary_codes = self._remove_nonprimary_code(aha_codes)
            for code in primary_codes:
                if code:
                    labels.add(code)

        return sorted(list(labels))

    def get_all_record_ids(self) -> List[str]:
        """Get all available record IDs without loading the actual data.

        Returns:
            List of record identifiers (ECG_IDs) as strings.
        """
        return self.metadata_df["ECG_ID"].astype(str).tolist()

    def get_stream(self) -> Generator[RawRecord, None, None]:
        """Stream raw records one at a time.

        Yields:
            RawRecord objects for each record in the dataset.
        """
        for record_id in self.get_all_record_ids():
            try:
                yield self.load_record(record_id)
            except Exception as e:
                logger.error(f"Error loading record {record_id}: {e}")
                continue

    def _extract_diagnosis_descs(self, codes: List[str]) -> List[str]:
        """Extract diagnostic descriptions from a list of AHA codes.

        Args:
            codes: List of AHA diagnostic codes.

        Returns:
            List of diagnostic descriptions.
        """
        descriptions = []
        for code in codes:
            try:
                desc_row = self.code_df[self.code_df["Code"].astype(int) == int(code)]
                if not desc_row.empty:
                    descriptions.append(desc_row.iloc[0]["Description"])
                else:
                    logger.warning(f"Description not found for code {code}")
                    descriptions.append("Unknown")
            except Exception:
                logger.warning(f"Error fetching description for code {code}")
                descriptions.append("Unknown")
        return descriptions

    @cache
    def load_record(self, record_id: str) -> RawRecord:
        """Load a single ECG record by its identifier.

        Args:
            record_id: The ECG record identifier.

        Returns:
            A RawRecord object with the ECG signal data, target labels, and metadata.
        """
        if record_id not in self.metadata_dict:
            raise ValueError(f"Record {record_id} not found in metadata")

        meta_row = self.metadata_dict[record_id]
        record_file = self.records_path / f"{record_id}.h5"
        if not record_file.exists():
            raise FileNotFoundError(
                f"H5 file for record {record_id} not found: {record_file}"
            )

        full_aha_codes = self._extract_aha_codes(str(meta_row["AHA_Code"]))
        primary_codes = self._remove_nonprimary_code(full_aha_codes)
        if not primary_codes:
            raise ValueError(f"Record {record_id} has no valid target labels")

        diagnosis_descriptions = self._extract_diagnosis_descs(primary_codes)

        data = self._read_ecg_h5(record_file)

        metadata = {
            "Patient_ID": meta_row["Patient_ID"],
            "Age": meta_row["Age"],
            "Sex": meta_row["Sex"],
            "N": meta_row["N"],
            "Date": meta_row["Date"],
            "diagnosis_descriptions": diagnosis_descriptions,
            "full_aha_codes": full_aha_codes,
            "file": str(record_file.relative_to(self.root)),
        }

        return RawRecord(
            id=record_id,
            data=data,
            target_labels=primary_codes,
            metadata=metadata,
        )

    @staticmethod
    def _extract_aha_codes(codes_str: str) -> List[str]:
        """Extract AHA diagnostic codes from a string.

        Args:
            codes_str: A string containing one or more AHA diagnostic codes.

        Returns:
            List of AHA diagnostic codes.
        """
        codes = codes_str.split(";")
        return [code.strip() for code in codes if code.strip()]

    # https://springernature.figshare.com/articles/dataset/Code_for_analyzing_the_ECG_records_and_dataset_splitting/17914088?backTo=/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802
    @staticmethod
    def _remove_nonprimary_code(x: List[str]) -> List[str]:
        """Extract primary diagnosis codes from a list of code groups.

        Each element in `x` is expected to be a string containing one or more codes
        separated by '+'. A code is considered 'primary' if its integer value is either
        less than 200 or greater than or equal to 500 (i.e. non-primary codes are in [200, 500)).

        Args:
            x: List of strings (each from splitting the raw AHA_Code field by ';').

        Returns:
            List of primary codes as strings (duplicates removed).
        """
        r = []
        for cx in x:
            for c in cx.split("+"):
                c = c.strip()
                if c:
                    try:
                        if int(c) < 200 or int(c) >= 500:
                            if c not in r:
                                r.append(c)
                    except ValueError:
                        # Skip codes that cannot be converted to integer.
                        continue
        return r

    # https://springernature.figshare.com/articles/dataset/Code_for_analyzing_the_ECG_records_and_dataset_splitting/17914088?backTo=/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802
    @staticmethod
    def _read_ecg_h5(path: Path) -> np.ndarray:
        """Read ECG signal data from an HDF5 file.

        Args:
            path: Path to the H5 file.

        Returns:
            A NumPy array of the ECG signal (converted to float32).
        """
        if not path.exists():
            raise FileNotFoundError(f"H5 file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            with h5py.File(str(path), "r") as h5f:
                keys = list(h5f.keys())
                if "ecg" not in keys:
                    raise ValueError(f"H5 file {path} contains no datasets")
                data = h5f["ecg"][()]
                data = np.array(data, dtype=np.float32)
                return data
        except Exception as e:
            raise RuntimeError(f"Error reading H5 file {path}: {e}") from e
