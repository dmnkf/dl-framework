import base64
import os
import tempfile
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.data.raw.data import RawRecord


def plot_processed_sample(
    data: Union[np.ndarray, RawRecord],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
) -> plt.Figure:
    """
    Visualize a processed CMR sample with ED, Mid-phase, and ES frames.

    Args:
        data: Either a 3D numpy array (height, width, frames) or RawRecord
             where frames correspond to [ED, Mid-phase, ES]
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    if isinstance(data, RawRecord):
        plot_data = data.data
        if title is None:
            title = f"Subject: {data.id}, Group: {data.target_labels}"
    else:
        plot_data = data

    assert (
        plot_data.ndim == 3 and plot_data.shape[2] == 3
    ), "Data must be 3D with exactly 3 frames (ED, Mid, ES)"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    frames_titles = ["End-Diastole (ED)", "Mid-Phase", "End-Systole (ES)"]

    for i, (ax, frame_title) in enumerate(zip(axes, frames_titles)):
        ax.imshow(plot_data[:, :, i], cmap="gray")
        ax.set_title(frame_title)
        ax.axis("off")

    if title:
        plt.suptitle(title, y=1.05)
    plt.tight_layout()
    return fig


def plot_raw_sample(
    data: np.ndarray,
    frame_indices: list[int],
    slice_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
) -> plt.Figure:
    """
    Visualize a raw 4D CMR sample at specific slice and frame indices.

    Args:
        data: 4D numpy array (height, width, slices, frames)
        frame_indices: List of frame indices [ED, Mid, ES] to display
        slice_idx: Slice index to use (default: middle slice)
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    assert data.ndim == 4, "Input data must be 4D (height, width, slices, frames)"
    assert len(frame_indices) == 3, "Must provide exactly 3 frame indices (ED, Mid, ES)"

    if slice_idx is None:
        slice_idx = data.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    frames_titles = ["End-Diastole (ED)", "Mid-Phase", "End-Systole (ES)"]

    for i, (ax, frame_title, frame_idx) in enumerate(
        zip(axes, frames_titles, frame_indices)
    ):
        ax.imshow(data[:, :, slice_idx, frame_idx], cmap="gray")
        ax.set_title(f"{frame_title}\nFrame {frame_idx}")
        ax.axis("off")

    if title:
        plt.suptitle(f"{title}, Slice: {slice_idx}", y=1.05)
    plt.tight_layout()
    return fig


def create_cardiac_cycle_animation(
    data: np.ndarray,
    frame_indices: list[int],
    slice_idx: Optional[int] = None,
    duration: float = 0.5,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> str:
    """
    Create an animated GIF of the cardiac cycle frames.

    Args:
        data: 4D numpy array (height, width, slices, frames)
        frame_indices: List of frame indices [ED, Mid, ES] to include
        slice_idx: Slice index to use (default: middle slice)
        duration: Duration for each frame in seconds
        title: Optional title for the animation
        figsize: Figure size (width, height)

    Returns:
        HTML string containing the embedded GIF
    """
    assert data.ndim == 4, "Input data must be 4D (height, width, slices, frames)"
    assert len(frame_indices) == 3, "Must provide exactly 3 frame indices (ED, Mid, ES)"

    if slice_idx is None:
        slice_idx = data.shape[2] // 2

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = data.min(), data.max()
    frames_titles = ["ED", "Mid-Phase", "ES"]

    def update(frame_idx):
        ax.clear()
        ax.imshow(
            data[:, :, slice_idx, frame_indices[frame_idx]],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        frame_title = f"{frames_titles[frame_idx]}\nFrame {frame_indices[frame_idx]}"
        if title:
            frame_title = f"{title}\n{frame_title}"
        ax.set_title(frame_title)
        ax.axis("off")

    ani = animation.FuncAnimation(
        fig, update, frames=range(3), interval=duration * 1000
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
        ani.save(tmpfile.name, writer="pillow", fps=1 / duration)
        with open(tmpfile.name, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(tmpfile.name)

    plt.close(fig)
    return f'<img src="data:image/gif;base64,{gif_base64}" alt="Cardiac Cycle GIF" />'


def plot_multi_slice_view(
    data: np.ndarray,
    frame_idx: int,
    num_slices: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create a grid view of multiple slices at a specific cardiac phase.

    Args:
        data: 4D numpy array (height, width, slices, frames)
        frame_idx: Frame index to display
        num_slices: Number of slices to display (default: all)
        figsize: Figure size (width, height)
    """
    assert data.ndim == 4, "Input data must be 4D (height, width, slices, frames)"

    total_slices = data.shape[2]
    if num_slices is None:
        num_slices = total_slices

    # Calculate grid dimensions
    n_cols = min(4, num_slices)
    n_rows = (num_slices + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    for idx, slice_idx in enumerate(slice_indices):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax.imshow(data[:, :, slice_idx, frame_idx], cmap="gray")
        ax.set_title(f"Slice {slice_idx}")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(num_slices, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig
