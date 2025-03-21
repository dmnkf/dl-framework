import json
import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

import torch
from tqdm import tqdm

from src.data.metadata_store import MetadataStore
from src.data.preprocessing.data import PreprocessedRecord
from src.data.raw.data import RawRecord, RawDataset

# Silence the SciPy warnings about byte ordering
warnings.filterwarnings("ignore", message="We do not support byte ordering")

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """Base class for data preprocessing pipelines."""

    def __init__(
        self,
        raw_data_handler: RawDataset,
        random_seed: Optional[int] = 1337,
        max_workers: Optional[int] = None,
        force_restart: bool = False,
        clean_interim: bool = False,
        metadata_store: Optional[MetadataStore] = None,
    ):
        self.clean_interim = clean_interim
        self.raw_data = raw_data_handler

        paths = self.raw_data.paths
        self.processed_dir = paths["processed"]
        self.interim_dir = paths["interim"]
        self.prepare_dirs()

        self.random_seed = random_seed
        self.max_workers = max_workers
        self.force_restart = force_restart

        self.metadata_store = metadata_store or MetadataStore(
            data_root=self.processed_dir
        )

    def prepare_dirs(self) -> None:
        """Create preprocessed directory and dataset info file."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def _get_processed_ids(self) -> Set[str]:
        """Get set of already processed sample IDs by scanning the interim directory."""
        if not self.interim_dir.exists():
            return set()

        return {
            f.stem
            for f in self.interim_dir.glob("*.pt")
            if f.is_file() and f.suffix == ".pt"
        }

    def _restart(self):
        """Remove all processed files and metadata."""
        logger.info("Starting cleanup of interim files...")
        self._cleanup_interim()
        logger.info("Interim files cleaned up. Resetting metadata store...")
        self.metadata_store.reset()
        logger.info("Metadata store reset completed.")

    def __save_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Save dataset info to disk."""
        with open(self.raw_data.paths["misc"]["dataset_info"], "w") as f:
            json.dump(dataset_info, f, indent=2)

    def process_record(self, record_id: str) -> Optional[PreprocessedRecord]:
        """Process a single record."""
        try:
            raw_record = self.raw_data.load_record(record_id)
            processed_record = self.preprocess_sample(raw_record)
            torch.save(processed_record, self.interim_dir / f"{processed_record.id}.pt")
            self.metadata_store.add(
                record_id=processed_record.id,
                metadata=processed_record.metadata,
                overwrite=self.force_restart,
            )
            return processed_record
        except (ValueError, FileNotFoundError) as e:
            # Log corrupted/missing files but don't stop processing
            logger.warning(f"Skipping corrupted/missing record {record_id}: {str(e)}")
            # Store metadata about the corrupted file
            self.metadata_store.add(
                record_id=record_id,
                metadata={"error": str(e), "status": "corrupted"},
                overwrite=True,
            )
            return None
        except Exception as e:
            # Re-raise unexpected errors
            logger.error(f"Unexpected error processing record {record_id}: {e}")
            raise

    def _preprocess_threaded(self, record_ids: List[str]) -> None:
        """Preprocess samples using a thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks upfront
            futures = [executor.submit(self.process_record, rid) for rid in record_ids]

            corrupted_count = 0
            with tqdm(
                total=len(futures),
                desc=f"Preprocessing: '{self.raw_data.dataset_key}'",
                unit="samples",
                dynamic_ncols=True,
                mininterval=0.5,  # More frequent updates
                position=0,
                leave=True,
            ) as pbar:
                # Process futures as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()  # Will be None for corrupted files
                        if result is None:
                            corrupted_count += 1
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        raise
                    pbar.update(1)

            if corrupted_count > 0:
                logger.warning(
                    f"Completed with {corrupted_count} corrupted/missing files"
                )

    def _preprocess_sequential(self, record_ids: List[str]) -> None:
        """Preprocess samples sequentially."""
        for record_id in tqdm(
            record_ids,
            desc=f"Preprocessing: '{self.raw_data.dataset_key}'",
            unit="samples",
            dynamic_ncols=True,
            mininterval=0.1,
            position=0,
            leave=True,
        ):
            self.process_record(record_id)

    def preprocess_all(self) -> None:
        """
        Run preprocessing on all samples, optionally in a thread pool.
        """
        self.__save_dataset_info(
            {
                "dataset_key": self.raw_data.dataset_key,
                "creation_date": datetime.now().isoformat(),
                "preprocessing": self.get_preprocessing_config(),
                "raw_metadata": self.raw_data.get_metadata(),
                "target_labels": self.raw_data.get_target_labels(),
            }
        )

        record_ids = self.raw_data.get_all_record_ids()
        logger.info(f"Found {len(record_ids)} total records to process")

        if self.force_restart:
            logger.info("Forcing restart of preprocessing pipeline.")
            self._restart()
            logger.info("Restart completed.")
        else:
            processed_ids = self._get_processed_ids()
            record_ids = [r for r in record_ids if r not in processed_ids]
            logger.info(
                f"Found {len(processed_ids)} already processed samples. "
                f"Processing {len(record_ids)} new samples."
            )

        try:
            if self.max_workers and self.max_workers > 1:
                logger.info(
                    f"Starting multi-threaded preprocessing with max_workers={self.max_workers}"
                )
                self._preprocess_threaded(record_ids)
            else:
                logger.info("Starting sequential preprocessing")
                self._preprocess_sequential(record_ids)

            if self.clean_interim:
                self._cleanup_interim()
        except Exception as e:
            logger.error(
                f"Error processing {self.raw_data.dataset_key} dataset: {e}",
                exc_info=True,
            )

    def _cleanup_interim(self) -> None:
        """Remove all interim files."""
        for f in self.interim_dir.glob("*.pt"):
            f.unlink()

    def preprocess_and_save_sample(self, record: RawRecord) -> None:
        """Preprocess a single sample and save it to disk."""
        processed_record = self.preprocess_sample(record)
        torch.save(processed_record, self.interim_dir / f"{record.id}.pt")

    @abstractmethod
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing-specific configuration."""
        pass

    @abstractmethod
    def preprocess_sample(self, raw_record: RawRecord) -> PreprocessedRecord:
        """Preprocess a single raw record sample."""
        pass
