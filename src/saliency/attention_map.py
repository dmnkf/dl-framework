import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib.collections import LineCollection

from src.models.ecg_classifier import ECGClassifier

SUPPORTED_MODELS = [ECGClassifier]


def is_supported_model(model):
    return model.__class__ in SUPPORTED_MODELS


def get_attention_layer(model):
    return model.blocks[-1].attn.attn_map


def get_attention_map(model, input_tensor):
    if not is_supported_model(model):
        raise ValueError(
            f"Model {model.__class__} not supported. Supported: {SUPPORTED_MODELS}"
        )

    with torch.no_grad():
        # Forward pass to populate attention maps
        _ = model(input_tensor)
        attention_maps = get_attention_layer(model)

    batch_figures = []
    for idx in range(input_tensor.shape[0]):
        fig = plot_attention(
            original_signal=input_tensor, attention_map=attention_maps, sample_idx=idx
        )
        batch_figures.append(fig)

    return batch_figures


def plot_attention(
    original_signal: torch.Tensor,
    attention_map: torch.Tensor,
    sample_idx: int,
    head_idx: int = 0,
):
    """
    Plot the attention as a heatmap over the signal.
    All channels of the signal for one head.

    Args:
        original_signal: Tensor of shape (B, C, C_sig, T_sig)
        attention_map: Tensor of shape (B, Heads, N, N) where N is num_patches + 1 (CLS token)
        sample_idx: Index of the sample to visualize
        head_idx: Index of the attention head to visualize (default: 0)
    """
    B, C, C_sig, T_sig = original_signal.shape
    B, Heads, N, N = attention_map.shape
    NpC = int((N - 1) / C_sig)  # Patches per channel

    # Normalize signal for visualization
    original_signal = original_signal + 0.5 * abs(original_signal.min())

    fig, axes = plt.subplots(nrows=C_sig, figsize=(16, 8))
    if C_sig == 1:
        axes = [axes]  # Make it iterable for single channel case

    for channel in range(C_sig):
        # Retrieve the attention of the channel
        attention_map_ch = attention_map[
            :, :, 1 + channel * NpC : 1 + (channel + 1) * NpC, 1:
        ]  # Ignore the cls token

        # Average the attention of all tokens to this channel
        attention_map_ch = attention_map_ch.mean(dim=-1)

        # Interpolate to match signal length
        attention_map_ch = F.interpolate(attention_map_ch, size=T_sig, mode="linear")

        # Get signal and attention for visualization
        original_signal_ch = original_signal[sample_idx, 0, channel].cpu()
        attention_map_ch = attention_map_ch[sample_idx].cpu()

        attn_min = attention_map_ch.min()
        attn_max = attention_map_ch.max()
        attention_map_ch = (attention_map_ch - attn_min) / (attn_max - attn_min + 1e-8)

        # Create time axis and vertices
        t = np.arange(T_sig)
        vertices = np.column_stack([t, original_signal_ch])

        # Plot signal
        axes[channel].plot(t, original_signal_ch, color="white", linewidth=2)

        # Create colored line segments based on attention weights
        segments = np.stack([vertices[:-1], vertices[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="YlGnBu",
            norm=plt.Normalize(
                attention_map_ch[head_idx].min(), attention_map_ch[head_idx].max()
            ),
            linewidth=1,
        )
        lc.set_array(attention_map_ch[head_idx])
        axes[channel].add_collection(lc)

        # Adjust plot aesthetics
        axes[channel].set_ylim(original_signal_ch.min(), original_signal_ch.max())
        if channel < C_sig - 1:
            axes[channel].set_xticks([])
        axes[channel].spines[["right", "top", "left", "bottom"]].set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(lc, cax=cbar_ax)
    cbar.set_label("Attention Weights")

    # Remove y labels
    [ax.yaxis.set_visible(False) for ax in axes]
    plt.tight_layout()
    return fig


# Adapted from https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/util/plot.py#L10
def plot_attention_old(original_signal, attentation_map, idx):
    """
    :input:
    original_signal (B, C, C_sig, T_sig)
    attention_map (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    Returns:
    matplotlib.figure.Figure - the generated figure
    """
    B, C, C_sig, T_sig = original_signal.shape
    B, Heads, N, N = attentation_map.shape

    NpC = int(N / C_sig)  # N_(C_sig)

    # only for nice visualization
    original_signal = original_signal + 0.5 * abs(original_signal.min())

    # (B, Heads, N_(C_sig), N_(C_sig)), attention map of the first signal channel
    channel = 0
    attentation_map = attentation_map[
        :,
        :,
        1 + (channel * NpC) : 1 + ((channel + 1) * NpC),
        1 + (channel * NpC) : 1 + ((channel + 1) * NpC),
    ]  # leave the cls token out
    # (B, Heads, N_(C_sig))
    attentation_map = attentation_map.mean(dim=2)
    attentation_map = F.normalize(attentation_map, dim=-1)
    attentation_map = attentation_map.softmax(dim=-1)
    # (B, Heads, T_sig)
    attentation_map = F.interpolate(attentation_map, size=T_sig, mode="linear")

    # (T_sig)
    original_signal = original_signal[idx, 0, 0].cpu()
    # (Heads, T_sig)
    attentation_map = attentation_map[idx].cpu()

    fig, axes = plt.subplots(nrows=Heads, sharex=True)

    for head in range(0, Heads):
        axes[head].plot(
            range(0, original_signal.shape[-1], 1), original_signal, zorder=2
        )  # (2500)
        sns.heatmap(
            attentation_map[head, :].unsqueeze(dim=0).repeat(15, 1),
            linewidth=0.5,  # (1, 2500)
            alpha=0.3,
            zorder=1,
            ax=axes[head],
        )
        axes[head].set_ylim(original_signal.min(), original_signal.max())

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()

    return fig
