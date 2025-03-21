# %% [code]
import sys
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path so that imports work correctly
project_root = Path().absolute().parent
sys.path.append(str(project_root))

from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

# For reproducibility
random.seed(42)
torch.manual_seed(42)

# %% [code]
data_root = Path(project_root) / "data"
acdc_data = UnifiedDataset(data_root, modality=DatasetModality.CMR, dataset_key="acdc")

record_idx = 18
raw = acdc_data.raw_dataset.get_nifti(acdc_data.get_all_record_ids()[record_idx])
processed = acdc_data[acdc_data.get_all_record_ids()[record_idx]].preprocessed_record

print("Raw CMR shape:", raw.shape)
print("Processed CMR shape:", processed.inputs.shape)

# %% [code]

# Raw CMR shape: (232, 256, 10, 30)
# plot raw frames
for i in range(raw.shape[-1]):
    slice = raw.shape[2] // 2
    plt.imshow(raw[:, :, slice, i], cmap="gray")
    plt.axis("off")
    plt.show()


# %% [code]

# Raw CMR shape: (232, 256, 10, 30)
# plot raw slices
for i in range(raw.shape[-2]):
    frame = raw.shape[3] // 2
    plt.imshow(raw[:, :, i, frame], cmap="gray")
    plt.axis("off")
    plt.show()


# %% [code]

# plot processed frames
for i in range(processed.inputs.shape[0]):
    plt.imshow(processed.inputs[i], cmap="gray")
    plt.axis("off")
    plt.show()


# %% [code]
data_root = Path(project_root) / "data"
ecg_data = UnifiedDataset(
    data_root, modality=DatasetModality.ECG, dataset_key="arrhythmia"
)

record_idx = 18
raw = ecg_data.raw_dataset.get_mat_signal(ecg_data.get_all_record_ids()[record_idx])
processed = ecg_data[ecg_data.get_all_record_ids()[record_idx]].preprocessed_record

print("Raw ECG shape:", raw.shape)
print("Processed ECG shape:", processed.inputs.shape)

# %% [code]

import numpy as np


def plot_ecg_signal(signal: np.ndarray, fs: int = 500) -> None:
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    time = np.arange(signal.shape[1]) / fs  # Convert samples to seconds

    fig, axes = plt.subplots(
        12, 1, figsize=(8, 10), sharex=True, constrained_layout=True
    )

    for i, ax in enumerate(axes):
        ax.plot(time, signal[i, :], color="black", linewidth=0.8)
        ax.set_ylabel(
            leads[i], rotation=0, labelpad=20, fontsize=10, verticalalignment="center"
        )
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    # save plot
    from datetime import datetime

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"ecg_signal_{current_time}.png")
    plt.show()


# %% [code]
plot_ecg_signal(raw)

# %% [code]
plot_ecg_signal(processed.inputs)

# %%
