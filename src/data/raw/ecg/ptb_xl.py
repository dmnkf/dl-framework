import ast
import logging
from functools import cache
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple

import numpy as np
import pandas as pd
import wfdb

from src.data.dataset import DatasetModality
from src.data.raw.data import RawDataset, RawRecord
from src.data.raw.registry import RawDatasetRegistry

logger = logging.getLogger(__name__)


@RawDatasetRegistry.register(DatasetModality.ECG.value, "ptbxl")
class PTBXL(RawDataset):
    """Handler for raw PTB-XL ECG data at 500 Hz.

    Expected data structure in the raw data directory (self.paths["raw"]):
        ptbxl/
        ├── ptbxl_database.csv         # Contains one row per record (indexed by ecg_id) with extensive metadata
        ├── scp_statements.csv         # Contains SCP-ECG annotation mappings (optional)
        └── records500/                # WFDB records at 500 Hz

    The ptbxl_database.csv file contains many columns including:
        - ecg_id, patient_id, age, sex, height, weight, nurse, site, device, recording_date, etc.
        - scp_codes: a string representation of a dictionary mapping SCP-ECG statement codes to likelihoods.
        - filename_hr: path (relative to the ptbxl folder) to the WFDB record for 500 Hz.
    """

    dataset_key = "ptbxl"
    modality = DatasetModality.ECG

    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Path to the dataset root folder.
                      It should contain the ptbxl folder with the PTB-XL files.
        """
        super().__init__(data_root)
        self.root = self.paths["raw"]

        self.database_csv = self.root / "ptbxl_database.csv"
        self.scp_csv = self.root / "scp_statements.csv"

        if not self.database_csv.exists():
            raise FileNotFoundError(f"Database CSV not found: {self.database_csv}")
        if not self.scp_csv.exists():
            logger.warning(f"SCP statements CSV not found: {self.scp_csv}")

        # Use "ecg_id" as the index for fast lookup.
        self.metadata_df = pd.read_csv(self.database_csv, index_col="ecg_id")

        # ecg_id is unique, but patient_id is not. Group by patient_id and use random record
        self.metadata_df = (
            self.metadata_df.groupby("patient_id")
            .sample(n=1, random_state=1337)
            .reset_index(drop=True)
        )

        self.metadata_df["patient_id"] = self.metadata_df["patient_id"].astype(int)
        self.metadata_df["scp_codes"] = self.metadata_df["scp_codes"].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str)
            else x  # scp codes are stored as dict strings
        )

        self.statements_dict = pd.read_csv(self.scp_csv, index_col=0).to_dict(
            orient="index"
        )

        self.metadata_dict: Dict[str, Any] = self.metadata_df.to_dict(orient="index")
        self.records_folder = (
            self.root / "records500"
        )  # we are only interested in the 500Hz data
        if not self.records_folder.exists():
            raise FileNotFoundError(f"Records folder not found: {self.records_folder}")

        self.filename_col = (
            "filename_hr"  # 500 Hz records / for 100 hz use "filename_lr"
        )

    def verify_data(self) -> None:
        """Verify the raw data structure and contents."""
        if not self.database_csv.exists():
            raise ValueError(f"Database CSV does not exist: {self.database_csv}")
        if not self.records_folder.exists():
            raise ValueError(f"Records folder does not exist: {self.records_folder}")

        for record_id in self.get_all_record_ids():
            file_path = self._get_file_path(record_id)
            header_path = file_path.with_suffix(".hea")
            dat_path = file_path.with_suffix(".dat")
            if not header_path.exists():
                raise FileNotFoundError(
                    f"WFDB header file for record {record_id} not found: {header_path}"
                )

            if not dat_path.exists():
                raise FileNotFoundError(
                    f"WFDB data file for record {record_id} not found: {dat_path}"
                )

        # based on https://physionet.org/content/ptb-xl/1.0.3/, we want only unique patients with one record per patient
        assert (
            len(self.metadata_df) == 18869
        ), f"Expected 18869 records, got {len(self.metadata_df)}"

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-specific metadata.

        Returns:
            A dictionary containing dataset details.
        """
        return {
            "num_leads": 12,
            "sampling_rate": 500,
            "record_length_seconds": 10,
            "total_records": len(self.metadata_df),
            "num_patients": int(self.metadata_df["patient_id"].nunique()),
        }

    def _get_file_path(self, record_id: str) -> Path:
        """Get the full path to the WFDB record file for a given record ID.

        Args:
            record_id: The unique ECG record identifier.

        Returns:
            Full path to the WFDB record file.
        """
        return self.root / self.metadata_dict[int(record_id)][self.filename_col]

    @cache
    def get_target_labels(self) -> List[str]:
        """Extract all unique SCP codes (with likelihood > 0) from the metadata.

        Returns:
            Sorted list of unique target label codes.
        """
        labels = set()
        for record_id in self.get_all_record_ids():
            codes = self._extract_scp_codes(int(record_id))
            labels.update(self._extract_target_labels(codes))
        return sorted(list(labels))

    def get_all_record_ids(self) -> List[str]:
        """Get all available record IDs without loading the actual data.

        Returns:
            List of record identifiers (ecg_id) as strings.
        """
        return self.metadata_df.index.astype(str).tolist()

    def get_stream(self) -> Generator[RawRecord, None, None]:
        """Stream raw records one at a time.

        Yields:
            RawRecord objects for each record.
        """
        for record_id in self.get_all_record_ids():
            try:
                yield self.load_record(record_id)
            except Exception as e:
                logger.error(f"Error loading record {record_id}: {e}")
                continue

    @cache
    def load_record(self, record_id: str) -> RawRecord:
        """Load a single ECG record by its identifier.

        Args:
            record_id: The unique ECG record identifier.

        Returns:
            A RawRecord object containing the ECG signal data, target labels, and metadata.
        """
        record_id = int(record_id)
        if record_id not in self.metadata_dict:
            raise ValueError(f"Record {record_id} not found in metadata.")
        meta = self.metadata_dict[record_id]

        file_path = self._get_file_path(record_id)
        record = self._read_wfdb_record(record_id)
        data = record[0]

        codes = self._extract_scp_codes(record_id)
        target_labels = self._extract_target_labels(codes)

        record_metadata = dict(meta)
        record_metadata["wfdb_file"] = str(file_path.relative_to(self.root))
        record_metadata["signal_fields"] = record[1]

        record_metadata["scp_statements"] = {
            code: self._extract_scp_statement(code) for code in codes
        }

        return RawRecord(
            id=record_id,
            data=data,
            target_labels=target_labels,
            metadata=record_metadata,
        )

    def _extract_scp_statement(self, code: str) -> str:
        """Extract the SCP statement for a given code from the SCP statements CSV.

        Args:
            code: SCP code to look up.

        Returns:
            SCP statement description.
        """
        if code in self.statements_dict:
            return self.statements_dict[code]
        raise ValueError(f"SCP code {code} not found in statements CSV.")

    def _extract_target_labels(self, scp_codes: Dict[str, float]) -> List[str]:
        """Extract target labels from the SCP codes dictionary.

        Args:
            scp_codes: Dictionary mapping SCP codes to likelihoods.

        Returns:
            List of target labels with likelihood > 0.
        """
        return [
            code for code, likelihood in scp_codes.items() if float(likelihood) >= 0
        ]  # include unknown codes (likelihood == 0): "... where likelihood is set to 0 if unknown) ..."

    def _extract_scp_codes(self, record_id: int) -> Dict[str, float]:
        """Extract SCP codes and likelihoods for a given record ID.

        Args:
            record_id: The unique ECG record identifier.

        Returns:
            Dictionary mapping SCP codes to likelihoods.
        """
        codes = self.metadata_dict[record_id].get("scp_codes")
        if not isinstance(codes, dict):
            raise ValueError(f"SCP codes not found for record {record_id}")
        return codes

    def _read_wfdb_record(self, record_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read the WFDB record data for a given record ID.

        Args:
            record_id: The unique ECG record identifier.

        Returns:
            Numpy array containing the ECG signal data.
        """
        file_path = self._get_file_path(record_id)
        try:
            record = wfdb.rdsamp(str(file_path))
            data = np.array(record[0], dtype=np.float32)
            data = np.swapaxes(data, 0, 1)
            return data, record[1]
        except Exception as e:
            raise RuntimeError(f"Error reading WFDB record {file_path}: {e}") from e
