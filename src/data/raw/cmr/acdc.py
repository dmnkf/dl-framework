import configparser
import logging
from functools import cache
from pathlib import Path
from typing import Dict, Any, List, Tuple
from typing import Generator

import nibabel as nib
import numpy as np

from src.data.raw.data import RawDataset, RawRecord
from src.data.raw.registry import RawDatasetRegistry
from src.data.dataset import DatasetModality

logger = logging.getLogger(__name__)


@RawDatasetRegistry.register("cmr", "acdc")
class ACDC(RawDataset):
    dataset_key = "acdc"
    modality = DatasetModality.CMR

    required_metadata_fields = {
        "ed_frame": None,
        "es_frame": None,
        "group": None,
        "height": None,
        "weight": None,
        "nb_frames": None,
    }

    def __init__(self, data_root: Path):
        super().__init__(data_root)
        self.data_path = self.paths["raw"]
        self.__nifti_files = {
            f.parent.name: f.absolute() for f in self.data_path.glob("**/*_4d.nii.gz")
        }
        self.__config_files = {
            f.parent.name: f.absolute() for f in self.data_path.glob("**/Info.cfg")
        }

    def verify_data(self) -> None:
        if not self.data_path.exists():
            raise ValueError(f"ACDC data path does not exist: {self.data_path}")

        if len(self.__nifti_files) == 0:
            raise ValueError(f"No 4D NIFTI files found in {self.data_path}")

        if len(self.__config_files) == 0:
            raise ValueError(f"No Info.cfg files found in {self.data_path}")

        # each NIFTI file should have a corresponding Info.cfg file
        nifti_stems = set(self.__nifti_files.keys())
        config_stems = set(self.__config_files.keys())
        if nifti_stems != config_stems:
            missing_nifti = nifti_stems - config_stems
            missing_config = config_stems - nifti_stems
            raise ValueError(
                f"Missing Info.cfg files for {missing_nifti} and NIFTI files for {missing_config}"
            )

        # based on https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
        assert (
            len(self.get_all_record_ids()) == 150
        ), f"Expected 150 records, got {len(self.get_all_record_ids())}"

    def get_metadata(self) -> Dict[str, Any]:
        return {"default_image_size": 256}

    def _get_record_paths(self, record_id: str) -> Tuple[Path, Path]:
        """Get paths to .nii.gz and Info.cfg files for a given record ID."""
        return self.__nifti_files.get(record_id), self.__config_files.get(record_id)

    @cache
    def get_target_labels(self) -> List[str]:
        """Get list of target labels using streaming to reduce memory usage."""
        groups = set()
        for record_id in self.get_all_record_ids():
            _, config_path = self._get_record_paths(record_id)
            group = self._parse_config(config_path).get("group")
            if group is not None:
                groups.add(group)
        return sorted(list(groups))

    def get_all_record_ids(self) -> List[str]:
        """Get all record IDs without loading data."""
        return list(self.__nifti_files.keys())

    def get_stream(self) -> Generator[RawRecord, None, None]:
        """Stream records one at a time to reduce memory usage."""
        for record_id in self.get_all_record_ids():
            yield self.load_record(record_id)

    @cache
    def load_record(self, record_id: str) -> RawRecord:
        nifti_file, config_file = self._get_record_paths(record_id)

        metadata = self._parse_config(config_file)
        frame_indices = self._calculate_frame_indices(metadata)
        image_data = self._get_extracted_frames(nifti_file, frame_indices)

        metadata.update(
            {
                f"{frame_type}_frame_idx": idx
                for frame_type, idx in zip(["ed", "mid", "es"], frame_indices)
            }
        )

        metadata["nifti_path"] = str(nifti_file.relative_to(self.data_path))

        group = metadata.pop("group", None)
        return RawRecord(
            id=record_id,
            data=image_data,
            target_labels=[group] if group else None,
            metadata=metadata,
        )

    @staticmethod
    def _calculate_frame_indices(metadata: Dict[str, Any]) -> List[int]:
        required_fields = ["ed_frame", "es_frame", "nb_frames"]
        if not all(metadata.get(field) is not None for field in required_fields):
            raise ValueError("Missing required frame information")

        ed_frame_idx = metadata["ed_frame"] - 1
        es_frame_idx = metadata["es_frame"] - 1
        nb_frames = metadata["nb_frames"]

        if ed_frame_idx <= es_frame_idx:
            mid_frame_idx = (ed_frame_idx + es_frame_idx) // 2
        else:
            mid_frame_idx = ((ed_frame_idx + es_frame_idx + nb_frames) // 2) % nb_frames

        return [ed_frame_idx, mid_frame_idx, es_frame_idx]

    def _get_extracted_frames(self, path: Path, frame_indices: List[int]) -> np.ndarray:
        nifti_data = self._read_nifti(path)
        slices = self._extract_middle_slice(nifti_data)
        return self._extract_frames(slices, frame_indices)

    @staticmethod
    def _read_nifti(path: Path) -> np.ndarray:
        try:
            return nib.load(str(path)).get_fdata()
        except Exception as e:
            raise RuntimeError(f"Error reading NIFTI file {path}: {e}")

    @staticmethod
    def _extract_middle_slice(data: np.ndarray) -> np.ndarray:
        """Gets the middle basal apical slice"""
        # We approximate the middle slice by taking the middle slice along the z-axis
        return data[:, :, data.shape[2] // 2, :]

    @staticmethod
    def _extract_frames(slice_data: np.ndarray, frame_indices: List[int]) -> np.ndarray:
        num_frames = slice_data.shape[-1]
        invalid_indices = [idx for idx in frame_indices if idx < 0 or idx >= num_frames]

        if invalid_indices:
            raise IndexError(
                f"Frame indices out of bounds: {invalid_indices}. "
                f"Valid range: [0, {num_frames - 1}]"
            )

        return slice_data[:, :, frame_indices]

    @cache
    def _parse_config(self, path: Path) -> Dict[str, Any]:
        """Parse config file with caching."""
        config = configparser.ConfigParser()
        with open(path, "r") as f:
            config.read_string("[DEFAULT]\n" + f.read())

        parsed_values = {
            "ed_frame": config["DEFAULT"].getint("ED"),
            "es_frame": config["DEFAULT"].getint("ES"),
            "group": config["DEFAULT"].get("Group"),
            "height": config["DEFAULT"].getfloat("Height"),
            "weight": config["DEFAULT"].getfloat("Weight"),
            "nb_frames": config["DEFAULT"].getint("NbFrame"),
        }
        return {k: v for k, v in parsed_values.items() if v is not None}
