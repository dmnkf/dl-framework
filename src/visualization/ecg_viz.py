import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_ecg_signals(
    ecg_data: np.ndarray, metadata: dict, sampling_rate: int = 500, duration: int = 10
):
    """
    Plot ECG signals for the given data and metadata.

    Args:
        ecg_data: 2D numpy array (leads, samples)
        metadata: Metadata containing patient information
        sampling_rate: Sampling rate of the ECG signal in Hz
        duration: Duration to plot in seconds
    """
    n_leads, n_samples = ecg_data.shape
    time = np.linspace(0, duration, duration * sampling_rate)

    fig, axes = plt.subplots(n_leads, 1, figsize=(15, 15), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time, ecg_data[i, : sampling_rate * duration], label=f"Lead {i+1}")
        ax.set_ylabel("Amplitude (μV)")
        ax.legend(loc="upper right")
        ax.grid(True)

    plt.xlabel("Time (seconds)")
    plt.suptitle(
        f"ECG Signals for Patient - {metadata.get('age', 'Unknown')} years, {'Male' if metadata.get('is_male', True) else 'Female'}",
        y=1.02,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
