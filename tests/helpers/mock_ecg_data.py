import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Literal


def create_mock_ecg_data(
    n_samples: int = 100,
    n_leads: int = 12,
    seq_length: int = 1000,
    add_noise: bool = True,
    noise_level: float = 0.1,
    include_baseline_wander: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[int]]:
    """Create mock ECG data with realistic dimensions and optional artifacts.

    Args:
        n_samples: Number of ECG samples to generate
        n_leads: Number of ECG leads (default 12 for standard ECG)
        seq_length: Length of each ECG sequence
        add_noise: Whether to add Gaussian noise
        noise_level: Standard deviation of the noise
        include_baseline_wander: Whether to add low-frequency baseline wander

    Returns:
        Tuple containing:
        - Dictionary mapping sample IDs to ECG tensors
        - List of sample IDs
        - List of labels (0-4 for different conditions)
    """
    data = {}
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    labels = np.random.randint(0, 5, n_samples).tolist()

    # Create time vector
    t = torch.linspace(0, 10, seq_length)

    for sample_id in sample_ids:
        # Base ECG-like signal with different frequencies for different components
        signal = torch.zeros(n_leads, seq_length)
        for lead in range(n_leads):
            # P wave
            p_wave = 0.3 * torch.sin(2 * torch.pi * 1 * t) * torch.exp(-((t - 1) ** 2))
            # QRS complex
            qrs = torch.exp(-((t - 3) ** 2)) * torch.sin(2 * torch.pi * 10 * t)
            # T wave
            t_wave = (
                0.5 * torch.sin(2 * torch.pi * 0.5 * t) * torch.exp(-((t - 5) ** 2))
            )

            # Combine components with slight variations per lead
            signal[lead] = p_wave + qrs + t_wave

            # Add lead-specific scaling
            signal[lead] *= 0.8 + 0.4 * torch.rand(1)

            if add_noise:
                noise = torch.randn_like(signal[lead]) * noise_level
                signal[lead] += noise

            if include_baseline_wander:
                baseline = 0.2 * torch.sin(2 * torch.pi * 0.1 * t)
                signal[lead] += baseline

        data[sample_id] = signal

    return data, sample_ids, labels


def create_mock_metadata(n_samples: int = 100) -> Dict[str, Dict]:
    """Create mock metadata for testing MetadataStore with realistic ECG metadata.

    Args:
        n_samples: Number of samples to generate metadata for

    Returns:
        Dictionary mapping sample IDs to metadata dictionaries
    """
    metadata = {}

    # Possible values for categorical fields
    diagnoses = ["normal", "afib", "svt", "vt", "sinus_brady"]
    genders = ["M", "F"]
    acquisition_devices = ["GE_MAC2000", "Phillips_TC50", "Mortara_ELI250"]

    for i in range(n_samples):
        sample_id = f"sample_{i}"
        metadata[sample_id] = {
            "patient": {
                "age": np.random.randint(18, 90),
                "gender": np.random.choice(genders),
                "weight": round(np.random.normal(70, 15), 1),
                "height": round(np.random.normal(170, 20), 1),
            },
            "acquisition": {
                "device": np.random.choice(acquisition_devices),
                "sample_rate": 500,
                "high_pass": 0.5,
                "low_pass": 40,
                "date": f"2023-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
            },
            "diagnosis": {
                "primary": np.random.choice(diagnoses),
                "confidence": round(np.random.uniform(0.6, 1.0), 2),
                "reviewed_by": f"doctor_{np.random.randint(1,5)}",
            },
            "signal_quality": {
                "noise_level": round(np.random.uniform(0, 0.3), 2),
                "baseline_wander": round(np.random.uniform(0, 0.2), 2),
                "leads_off": [],
            },
        }

    return metadata


def create_mock_ecg_batch(
    batch_size: int = 32, n_leads: int = 12, seq_length: int = 1000, device: str = "cpu"
) -> torch.Tensor:
    """Create a mock batch of ECG data for testing preprocessing functions.

    Args:
        batch_size: Number of ECG samples in the batch
        n_leads: Number of ECG leads
        seq_length: Length of each ECG sequence
        device: Device to place the tensor on

    Returns:
        Tensor of shape (batch_size, n_leads, seq_length)
    """
    batch = torch.randn(batch_size, n_leads, seq_length, device=device)

    # Add some structure to make it more ECG-like
    t = torch.linspace(0, 10, seq_length, device=device)
    qrs_complex = torch.exp(-((t - 5) ** 2)) * torch.sin(2 * torch.pi * 10 * t)

    # Add QRS-like pattern to each sample with variations
    for i in range(batch_size):
        scale = 0.8 + 0.4 * torch.rand(n_leads, 1, device=device)
        batch[i] += scale * qrs_complex

    return batch
