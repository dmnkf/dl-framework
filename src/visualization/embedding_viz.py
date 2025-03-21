"""
Visualization utilities for embedding analysis and visualization.
This module provides functions for dimensionality reduction, group analysis,
and visualization of embeddings with various plotting utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import Dict, List, Optional, Union
from pathlib import Path


def run_umap(
    embeddings, metric="euclidean", min_dist=0.1, random_state=42, n_neighbors=15, n_components=2
):
    reducer = umap.UMAP(
        metric=metric,
        min_dist=min_dist,
        random_state=random_state,
        n_neighbors=n_neighbors,
        n_components=n_components,
    )
    embedding_2d = reducer.fit_transform(embeddings)
    return embedding_2d


def extract_integration_name_for_group(labels_meta, group_name):
    """
    Given a list of label dictionaries and a target group name,
    return the integration name of the first label that matches the group.
    If no label matches, return None.
    """
    for label in labels_meta:
        if label.get("group") == group_name:
            return label.get("integration_name")
    return None


def extract_labels_for_group(labels_meta, group_name, single_label_only=False):
    """
    Given a list of label dictionaries and a target group name,
    return all labels that match the group.

    Args:
        labels_meta (list): List of label dictionaries
        group_name (str): Name of the target group
        single_label_only (bool): If True, only return labels if there's exactly one label in the group

    Returns:
        list: List of matching label dictionaries, or empty list if no matches or
             if single_label_only=True and multiple labels exist
    """
    matching_labels = [
        label for label in labels_meta if label.get("group") == group_name
    ]

    if single_label_only and len(matching_labels) != 1:
        return []

    return matching_labels


def prepare_group_data(df, group_name, max_samples=100):
    """
    For a given disease group, filter records that have EXACTLY one label with 'group_name'
    (but can have other labels in different groups),
    then store the other integration names for multi-labeled records.
    """
    def has_exact_one_in_group(labels_meta):
        count = sum(lbl.get("group") == group_name for lbl in labels_meta)
        return count == 1

    df_group = df[df["labels_meta"].apply(has_exact_one_in_group)].copy()

    # Single integration_name for this group
    df_group["integration_name"] = df_group["labels_meta"].apply(
        lambda lm: extract_integration_name_for_group(lm, group_name)
    )
    df_group = df_group.dropna(subset=["integration_name"])

    # total label count
    df_group["total_labels"] = df_group["labels_meta"].apply(len)

    # Identify other integration names if multi-labeled
    def get_other_integration_names(labels_meta):
        """
        Return integration names for labels that do NOT belong to `group_name`.
        """
        others = []
        for lbl in labels_meta:
            if lbl.get("group") != group_name:
                iname = lbl.get("integration_name", "Unknown")
                others.append(iname)
        return sorted(set(others))

    df_group["other_inames"] = df_group["labels_meta"].apply(get_other_integration_names)

    # Mark "exclusive" vs "multi" based on whether there are other integration names
    def is_multi(other_inames):
        return "multi" if len(other_inames) > 0 else "exclusive"

    df_group["mlabel_flag"] = df_group["other_inames"].apply(is_multi)

    # sort by total_labels asc, then sample
    df_group_sorted = df_group.sort_values("total_labels")
    df_group_sub = df_group_sorted.head(max_samples)
    return df_group, df_group_sub

def plot_umap_for_group_extended(
    global_umap, df_global, df_group_sub, group_name, save_path=None
):
    """
    For each record in df_group_sub, we know:
      - 'integration_name': the single label in 'group_name'
      - 'mlabel_flag': "exclusive" or "multi"
      - 'other_inames': list of integration names for other groups
    We'll color by integration_name, use 'o' vs 'x' for exclusive vs multi,
    and annotate multi-labeled points with the other integration labels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    unique_inames = sorted(df_group_sub["integration_name"].unique())
    palette = sns.color_palette("hsv", len(unique_inames))
    color_map = dict(zip(unique_inames, palette))

    plt.figure(figsize=(10, 8))

    # Background
    plt.scatter(
        global_umap[:, 0],
        global_umap[:, 1],
        color="lightgray",
        marker="o",
        edgecolor="none",
        s=50,
        alpha=0.5,
        label="_nolegend_",
    )

    for i in df_group_sub.index:
        iname = df_group_sub.loc[i, "integration_name"]
        ccol = color_map[iname]
        marker_flag = df_group_sub.loc[i, "mlabel_flag"]  # "exclusive" or "multi"
        marker = "o" if marker_flag == "exclusive" else "x"
        x, y = global_umap[i, 0], global_umap[i, 1]

        plt.scatter(
            x,
            y,
            color=ccol,
            marker=marker,
            edgecolor="k",
            s=80,
            alpha=0.9,
        )

        # If multi-labeled, show the other integration labels
        if marker_flag == "multi":
            other_inames = df_group_sub.loc[i, "other_inames"]  # list of strings
            anno_text = ", ".join(other_inames)
            # offset text slightly so it doesn't overlap
            plt.text(x+0.5, y+0.5, anno_text, fontsize=8, color=ccol)

    plt.title(f"{group_name}: exactly 1 label in this group + possibly other integration labels")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Legend for integration_name
    legend_handles = []
    for nm, col in color_map.items():
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=8,
                label=nm,
            )
        )
    # Marker shapes legend
    marker_legend = [
        plt.Line2D([], [], marker="o", color="k", label="Exclusive", linestyle="None"),
        plt.Line2D([], [], marker="x", color="k", label="Multi-labeled", linestyle="None"),
    ]
    plt.legend(handles=legend_handles+marker_legend, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_umap_for_group(
    global_umap, df_global, df_group_sub, group_name, save_path=None
):
    # For coloring, determine unique integration names in the subsampled group data
    unique_names = sorted(df_group_sub["integration_name"].unique())
    palette = sns.color_palette("hsv", len(unique_names))
    color_map = {name: palette[i] for i, name in enumerate(unique_names)}

    plt.figure(figsize=(10, 8))

    # Plot global background (all records) in light gray
    global_idx = df_global.index.tolist()
    plt.scatter(
        global_umap[global_idx, 0],
        global_umap[global_idx, 1],
        color="lightgray",
        marker="o",
        edgecolor="none",
        s=50,
        alpha=0.5,
        label="_nolegend_",
    )

    # Overlay highlighted records for the group using the integration name for coloring and marker style based on label count
    highlight_idx = df_group_sub.index.tolist()
    for i in highlight_idx:
        name = df_group_sub.loc[i, "integration_name"]
        # Use "x" marker if record has multiple labels, otherwise "o"
        marker = "x" if df_group_sub.loc[i, "n_labels"] > 1 else "o"
        plt.scatter(
            global_umap[i, 0],
            global_umap[i, 1],
            color=color_map[name],
            marker=marker,
            edgecolor="k",
            s=80,
            alpha=0.9,
        )

    plt.title(
        f"Global UMAP Projection (Fine-Tuned) - Group: {group_name}\nHighlighted by Integration Name"
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Create legend handles for integration names (color legend)
    handles_color = []
    for name, col in color_map.items():
        handles_color.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=8,
                label=name,
            )
        )

    # Create legend handles for marker styles
    handles_marker = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="k",
            markersize=8,
            label="Single-label",
            linestyle="None",
        ),
        plt.Line2D(
            [],
            [],
            marker="x",
            color="k",
            markersize=8,
            label="Multi-label",
            linestyle="None",
        ),
    ]

    # Combine the legends
    handles = handles_color + handles_marker
    plt.legend(
        handles=handles,
        title="Integration Name & Label Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_integration_name_distribution(df_group, group_name, save_path=None):
    # Use the same color mapping logic as in plot_umap_for_group
    unique_names = sorted(df_group["integration_name"].unique())
    palette = sns.color_palette("hsv", len(unique_names))
    color_map = {name: palette[i] for i, name in enumerate(unique_names)}
    
    # Get counts and create the plot
    counts = df_group["integration_name"].value_counts()
    # Sort by name to match UMAP plot colors
    counts = counts.reindex(unique_names)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette=color_map)
    plt.title(f"Integration Name Distribution in Group: {group_name}")
    plt.xlabel("Integration Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
