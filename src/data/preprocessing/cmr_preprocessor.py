from typing import Dict, Any

import numpy as np
import torch

from src.data.preprocessing.base_preprocessor import (
    BasePreprocessor,
    PreprocessedRecord,
)
from src.data.raw.data import RawDataset, RawRecord


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities to [0, 1] range.

    Args:
        image: Input image array

    Returns:
        Normalized image array
    """
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


class CMRPreprocessor(BasePreprocessor):
    """Preprocessor for CMR data."""

    def __init__(
        self,
        raw_data_handler: RawDataset,
        image_size: int = 256,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        super(CMRPreprocessor, self).__init__(raw_data_handler, *args, **kwargs)
        self.image_size = image_size
        self.normalize = normalize

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing-specific configuration."""
        return {"image_size": self.image_size, "normalize": self.normalize}

    def preprocess_sample(self, raw_record: RawRecord) -> PreprocessedRecord:
        """Preprocess a single CMR record."""
        image_data = raw_record.data

        assert image_data.ndim == 3, "Expected 3D CMR data: (height, width, slices)"
        assert image_data.shape[-1] == 3, "Expected 3 slices in the input data"

        curr_height, curr_width = image_data.shape[0:2]
        target_size = 210
        processed_slices = np.zeros((target_size, target_size, 3))

        pad_height = max(0, target_size - curr_height)
        pad_width = max(0, target_size - curr_width)
        crop_height = max(0, curr_height - target_size)
        crop_width = max(0, curr_width - target_size)

        h_start = crop_height // 2
        w_start = crop_width // 2
        pad_h_start = pad_height // 2
        pad_w_start = pad_width // 2

        src_h_slice = slice(h_start, curr_height - crop_height + h_start)
        src_w_slice = slice(w_start, curr_width - crop_width + w_start)
        dst_h_slice = slice(pad_h_start, target_size - pad_height + pad_h_start)
        dst_w_slice = slice(pad_w_start, target_size - pad_width + pad_w_start)

        processed_slices[dst_h_slice, dst_w_slice, :] = image_data[
            src_h_slice, src_w_slice, :
        ]

        processed_slices = np.transpose(processed_slices, (2, 0, 1))

        if self.normalize:
            for i in range(processed_slices.shape[0]):
                processed_slices[i] = normalize_image(processed_slices[i])

        return PreprocessedRecord(
            id=raw_record.id,
            inputs=torch.tensor(processed_slices, dtype=torch.float32),
            target_labels=raw_record.target_labels,
            metadata=raw_record.metadata,
        )
