from typing import Dict, Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.data.preprocessing.base_preprocessor import (
    BasePreprocessor,
    PreprocessedRecord,
)
from src.data.raw.data import RawDataset, RawRecord


# https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L16
class Normalisation(object):
    """
    Time series normalisation.
    """

    def __init__(self, mode="group_wise", groups=[3, 6, 12]) -> None:
        self.mode = mode
        self.groups = groups

    def __call__(self, sample) -> np.array:
        sample_dtype = sample.dtype
        if self.mode == "sample_wise":
            mean = np.mean(sample)
            var = np.var(sample)

        elif self.mode == "channel_wise":
            mean = np.mean(sample, axis=-1, keepdims=True)
            var = np.var(sample, axis=-1, keepdims=True)

        elif self.mode == "group_wise":
            mean = []
            var = []
            lower_bound = 0
            for idx in self.groups:
                mean_group = np.mean(
                    sample[lower_bound:idx], axis=(0, 1), keepdims=True
                )
                mean_group = np.repeat(
                    mean_group, repeats=int(idx - lower_bound), axis=0
                )
                var_group = np.var(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                var_group = np.repeat(var_group, repeats=int(idx - lower_bound), axis=0)
                lower_bound = idx
                mean.extend(mean_group)
                var.extend(var_group)
            mean = np.array(mean, dtype=sample_dtype)
            var = np.array(var, dtype=sample_dtype)
        normalised_sample = (sample - mean) / (var + 1.0e-12) ** 0.5
        return normalised_sample


# https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L58
def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Asymmetric Least Squares Smoothing, i.e. asymmetric weighting of deviations to correct a baseline
    while retaining the signal peak information.
    Refernce: Paul H. C. Eilers and Hans F.M. Boelens, Baseline Correction with Asymmetric Least Squares Smoothing (2005).
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


class ECGPreprocessor(BasePreprocessor):
    """Preprocessor for ECG data."""

    def __init__(
        self,
        raw_data_handler: RawDataset,
        sequence_length: int = 5000,  # 10 seconds @ 500Hz
        normalize_mode: str = "group_wise",
        *args,
        **kwargs,
    ):
        super(ECGPreprocessor, self).__init__(raw_data_handler, *args, **kwargs)
        self.sequence_length = sequence_length
        self.normalize_mode = normalize_mode

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing-specific configuration."""
        return {
            "sequence_length": self.sequence_length,
            "normalize_mode": self.normalize_mode,
        }

    def preprocess_sample(self, raw_record: RawRecord) -> PreprocessedRecord:
        """Preprocess a single ECG record."""
        raw_signal = raw_record.data
        assert raw_signal.ndim == 2, "Expected 2D signal data: (channels, samples)"
        assert raw_signal.shape[0] == 12, "Expected 12 ECG leads"
        # note: we have a hard assumption, that the sampling rate is 500Hz. this is currently not code-asserted

        # truncate if longer than expected
        num_samples = raw_signal.shape[1]
        if num_samples > self.sequence_length:
            # Truncate the signal if it is longer than expected.
            raw_signal = raw_signal[:, : self.sequence_length]
        elif num_samples < self.sequence_length:
            # we do not support upsamling or padding
            raise ValueError(
                f"Signal duration is shorter than expected: {num_samples} samples < {self.sequence_length}"
            )

        # https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L78
        np.nan_to_num(raw_signal, copy=False)

        # https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L81
        signal_std = np.std(raw_signal)
        np.clip(raw_signal, a_min=-4 * signal_std, a_max=4 * signal_std, out=raw_signal)

        # https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L85
        baselines = np.zeros_like(raw_signal)
        for lead in range(raw_signal.shape[0]):
            baselines[lead] = baseline_als(raw_signal[lead], lam=1e7, p=0.3, niter=5)
        raw_signal -= baselines

        # https://github.com/oetu/MMCL-ECG-CMR/blob/674fa220efd098d30eee0fd72ed0b626046cea84/utils/preprocessing.py#L91
        transform = Normalisation(mode=self.normalize_mode, groups=[3, 6, 12])
        processed_signal = transform(raw_signal)

        assert processed_signal.shape == (12, self.sequence_length), (
            f"Expected processed signal shape: (12, {self.sequence_length}), "
            f"got: {processed_signal.shape}"
        )

        return PreprocessedRecord(
            id=raw_record.id,
            inputs=torch.tensor(processed_signal, dtype=torch.float32),
            target_labels=raw_record.target_labels,
            metadata=raw_record.metadata,
        )
