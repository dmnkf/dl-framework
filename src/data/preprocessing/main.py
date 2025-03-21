#!/usr/bin/env python3

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import rootutils

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

import src.utils.utils as utils
from src.data.raw.registry import RawDatasetRegistry
from src.data.preprocessing.cmr_preprocessor import CMRPreprocessor
from src.data.preprocessing.ecg_preprocessor import ECGPreprocessor


def setup_logging(log_file: str) -> None:
    """
    Configure logging to output to both a file and the console.

    The file handler logs all levels (DEBUG and above), while the console handler logs
    messages at the level specified by utils.get_log_level() (typically INFO and above).
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    log_level = utils.get_log_level()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s"
    )

    file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


setup_logging(str(PROJECT_ROOT / "logs" / "preprocessing.log"))
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for processing a single dataset."""
    parser = argparse.ArgumentParser(
        description="Preprocess a single dataset (ECG or CMR) identified by a unique key."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing raw data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Unique dataset key to process",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing. If not specified, processing is sequential.",
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Force restart preprocessing from scratch, ignoring any existing processed files",
    )
    parser.add_argument(
        "--clean_interim",
        action="store_true",
        help="Clean interim directory after processing",
    )
    # ECG-specific arguments (used only if the dataset belongs to the ECG modality)
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5000,
        help="Desired sequence length for the processed ECG signals",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="group_wise",
        choices=["sample_wise", "channel_wise", "group_wise"],
        help="Normalization strategy to use for ECG signals",
    )
    # CMR-specific arguments (used only if the dataset belongs to the CMR modality)
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Target size for resizing CMR images",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply intensity normalization for CMR images",
    )

    return parser.parse_args()


def process_dataset(args: argparse.Namespace) -> None:
    """
    Process a single dataset identified by a unique key.
    """
    abs_data_root = Path(args.data_root).absolute()

    available_datasets = RawDatasetRegistry.list_datasets()
    modality_found = None

    for modality, datasets in available_datasets.items():
        if args.dataset in datasets:
            modality_found = modality
            break

    if modality_found is None:
        logger.error("Dataset key '%s' not found.", args.dataset)
        logger.info("Available datasets:")
        for modality, datasets in available_datasets.items():
            logger.info("%s:", modality.upper())
            for ds in datasets:
                logger.info("  - %s", ds)
        sys.exit(1)

    logger.info(
        "Processing dataset '%s' for modality '%s'.",
        args.dataset,
        modality_found.upper(),
    )
    try:
        raw_data_class = RawDatasetRegistry.get_handler(modality_found, args.dataset)
        raw_data = raw_data_class(data_root=abs_data_root)
        raw_data.verify_data()

        if modality_found == "ecg":
            preprocessor = ECGPreprocessor(
                raw_data_handler=raw_data,
                sequence_length=args.sequence_length,
                normalize_mode=args.normalize_mode,
                max_workers=args.max_workers,
                force_restart=args.force_restart,
                clean_interim=args.clean_interim,
            )
        elif modality_found == "cmr":
            preprocessor = CMRPreprocessor(
                raw_data_handler=raw_data,
                image_size=args.image_size,
                normalize=args.normalize,
                max_workers=args.max_workers,
                force_restart=args.force_restart,
                clean_interim=args.clean_interim,
            )
        else:
            raise ValueError(f"Unknown modality: {modality_found}")

        preprocessor.preprocess_all()
        logger.info("Finished processing dataset '%s'.", args.dataset)
    except Exception as e:
        logger.error(
            "Error processing dataset '%s': %s", args.dataset, e, exc_info=True
        )
        raise


def main() -> None:
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
