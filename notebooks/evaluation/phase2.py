#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Subcluster Analysis within Single-Label Groups
-------------------------------------------------------
UMAP on the FULL test set (single + multi-labeled),
focusing on records with two labels where one is a label of interest.
This helps identify potential subclustering within single-label groups.
"""

# %% [markdown]
# # Phase 2 - Subcluster Analysis Notebook
# In this notebook-style script, we analyze subclusters within specific **single-labeled** groups,
# focusing on dual-labeled records that share a primary label of interest.

# %% [markdown]
# ## Imports

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN

# Ensure project path is in sys.path
project_root = Path().absolute().parent.parent
sys.path.append(str(project_root))

from src.visualization.embedding_viz import run_umap
from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 12


# %% [markdown]
# ## Color Palettes and Helpers

# %%
# Define the color palette exactly as in phase1.py
SINGLE_COLOR_PALETTE = sns.color_palette("colorblind", 5)

FIXED_LABEL_COLORS = {
    "SR": SINGLE_COLOR_PALETTE[0], 
    "SB": SINGLE_COLOR_PALETTE[1],  
    "AFIB": SINGLE_COLOR_PALETTE[2],
    "GSVT": SINGLE_COLOR_PALETTE[3], 
    "PACE": SINGLE_COLOR_PALETTE[4],  
}


# For secondary labels, use a more separated palette
# 'Paired' has good contrast between adjacent colors
DUAL_COLOR_PALETTE = sns.color_palette("Paired", 12)

# Ensure consistency by using the same palette for all groups
COLOR_PALETTES = {
    "Rhythm": SINGLE_COLOR_PALETTE,
    "Morphology": SINGLE_COLOR_PALETTE,
    "Duration": SINGLE_COLOR_PALETTE,
    "Amplitude": SINGLE_COLOR_PALETTE,
    "Other": SINGLE_COLOR_PALETTE,
}

GROUP_MARKERS = {
    "Rhythm": "o",
    "Morphology": "s",
    "Duration": "^",
    "Amplitude": "D",
    "Other": "X",
}

LABELS_OF_INTEREST = ["SR", "AFIB", "SB", "GSVT", "PACE"]
LABELS_OF_INTEREST.sort()

# %% [markdown]
# ## Data Loading & Metadata Extraction

# %%
print("Phase 2: Analyzing subclustering within single-label groups.")
arr_data = UnifiedDataset(
    Path(project_root) / "data", modality=DatasetModality.ECG, dataset_key="arrhythmia"
)
arr_splits = arr_data.get_splits()
arr_test_ids = arr_splits.get("test", [])
arr_md_store = arr_data.metadata_store

# Extract demographic metadata for all test records
demographic_data = {}
for rid in arr_test_ids:
    meta = arr_md_store.get(rid, {})
    age = meta.get("age", None)
    is_male = meta.get("is_male", None)
    demographic_data[rid] = {
        "age": age if isinstance(age, (int, float)) else None,
        "is_male": is_male if isinstance(is_male, bool) else None,
    }
print(f"Extracted demographic data for {len(demographic_data)} records")

pretrained_embedding = "baseline"
finetuned_embedding = "fine_tuned_50"


def extract_extended_records_info(arr_test_ids, arr_md_store):
    records_info = []
    emb_base_list = []
    emb_ft_list = []
    for rid in arr_test_ids:
        meta = arr_md_store.get(rid, {})
        labels_meta = meta.get("labels_metadata", [])
        age = meta.get("age", None)
        is_male = meta.get("is_male", None)
        age = age if isinstance(age, (int, float)) else None
        is_male = is_male if isinstance(is_male, bool) else None
        try:
            emb_base = arr_data.get_embeddings(
                rid, embeddings_type=pretrained_embedding
            )
            emb_ft = arr_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
        except Exception as e:
            print(f"Skipping {rid} (missing embeddings). Err: {e}")
            continue
        records_info.append(
            {
                "record_id": rid,
                "labels_meta": labels_meta,
                "n_labels": len(labels_meta),
                "age": age,
                "is_male": is_male,
            }
        )
        emb_base_list.append(emb_base)
        emb_ft_list.append(emb_ft)
    return records_info, emb_base_list, emb_ft_list


records_info, emb_base_list, emb_ft_list = extract_extended_records_info(
    arr_test_ids, arr_md_store
)
if not records_info:
    print("No records found. Exiting.")
    sys.exit()

df_records = pd.DataFrame(records_info)
df_records["row_idx"] = df_records.index
baseline_embeddings = np.vstack(emb_base_list)
finetuned_embeddings = np.vstack(emb_ft_list)

print(f"Total records loaded: {len(df_records)}")
print(" - Baseline shape:", baseline_embeddings.shape)
print(" - Fine-tuned shape:", finetuned_embeddings.shape)
print(f" - With age data: {df_records['age'].notna().sum()}")
print(f" - With sex data: {df_records['is_male'].notna().sum()}")

# %% [markdown]
# ## UMAP Projection

# %%
# Run UMAP on the full dataset
print("Running UMAP on baseline embeddings...")
baseline_umap = run_umap(baseline_embeddings, n_neighbors=30, min_dist=0.25, random_state=42)
print("Running UMAP on fine-tuned embeddings...")
finetuned_umap = run_umap(finetuned_embeddings, n_neighbors=30, min_dist=0.25, random_state=42)

# Add UMAP coordinates to the dataframe
df_records["umap_base_x"] = baseline_umap[:, 0]
df_records["umap_base_y"] = baseline_umap[:, 1]
df_records["umap_ft_x"] = finetuned_umap[:, 0]
df_records["umap_ft_y"] = finetuned_umap[:, 1]

# %% [markdown]
# ## Prepare Single-Labeled and Dual-Labeled Records

# %%
# Helper function to check if a record has a label of interest
def has_label_of_interest(labels_meta):
    for lm in labels_meta:
        if lm["integration_name"] in LABELS_OF_INTEREST:
            return True
    return False

# Helper function to count labels of interest in a record
def count_labels_of_interest(labels_meta):
    count = 0
    for lm in labels_meta:
        if lm["integration_name"] in LABELS_OF_INTEREST:
            count += 1
    return count

# Get single-labeled records with a label of interest
df_single_rhythm = df_records[
    (df_records["n_labels"] == 1)
    & (df_records["labels_meta"].apply(has_label_of_interest))
].copy()

# Get dual-labeled records where exactly one label is a label of interest
# and the other label is NOT a label of interest
df_dual_rhythm = df_records[
    (df_records["n_labels"] == 2)
    & (df_records["labels_meta"].apply(count_labels_of_interest) == 1)
].copy()

# Get records with no label of interest
df_no_rhythm = df_records[
    ~df_records["labels_meta"].apply(has_label_of_interest)
].copy()

# %% [markdown]
# ## Rhythm Label Analysis: Single vs. Dual-Labeled Records
# 
# In this section, we analyze whether the Rhythm label is the primary driver of clustering in the UMAP space.
# We'll identify records with exactly one Rhythm label plus one other label, and compare their positions
# to single-labeled Rhythm records.

# %%
def has_single_rhythm_label(labels_meta):
    """
    Returns True if exactly one label is from the 'Rhythm' group,
    and there are exactly 2 labels total.
    """
    if len(labels_meta) != 2:
        return False
    rhythm_count = sum(1 for lbl in labels_meta if lbl.get("group") == "Rhythm")
    return rhythm_count == 1

# Filter for records with exactly one Rhythm label and one other label
# df_dual_rhythm = df_records[df_records["labels_meta"].apply(has_single_rhythm_label)].copy()
# print(f"Rhythm + 1-other-label records: {len(df_dual_rhythm)}")

# Filter for records with exactly one label from the Rhythm group
# df_single_rhythm = df_records[
#     (df_records["n_labels"] == 1)
#     & (df_records["labels_meta"].apply(lambda lm: lm[0]["group"] == "Rhythm"))
# ].copy()
# print(f"Single-labeled Rhythm records: {len(df_single_rhythm)}")

# %%
# Extract Rhythm and Other labels for each record
def get_rhythm_label(row):
    """
    Get the Rhythm label from a record.
    Special case: PACE is included even if it's in the Morphology group.
    """
    for label_meta in row["labels_meta"]:
        label_name = label_meta.get("integration_name", "")
        # Include other rhythm labels
        if label_name in LABELS_OF_INTEREST:
            return label_name
    return None

def get_other_label(row):
    """
    Get the first non-Rhythm label from a record.
    Special case: PACE is considered a Rhythm label even if it's in the Morphology group.
    """
    for label_meta in row["labels_meta"]:
        label_name = label_meta.get("integration_name", "")
        # Skip PACE and other rhythm labels
        if label_name == "PACE" or label_name in LABELS_OF_INTEREST:
            continue
        return label_name
    return None

# Add rhythm label to both dataframes
df_dual_rhythm["rhythm_label"] = df_dual_rhythm.apply(get_rhythm_label, axis=1)
df_single_rhythm["rhythm_label"] = df_single_rhythm.apply(get_rhythm_label, axis=1)

# Add the other label to the dual-labeled records
df_dual_rhythm["other_label"] = df_dual_rhythm.apply(get_other_label, axis=1)

# Add a placeholder for single-labeled records (they have no other label)
df_single_rhythm["other_label"] = None

# Combine datasets for analysis
subset_df = pd.concat([df_dual_rhythm, df_single_rhythm], ignore_index=True)

# %% [markdown]
# ## Quantitative Analysis: Silhouette Score Comparison

# %%
from sklearn.metrics import silhouette_score

# Prepare data for silhouette score calculation
subset_df["rhythm_cluster"] = subset_df["rhythm_label"]
subset_df["other_cluster"] = subset_df["other_label"]

# Calculate silhouette scores for baseline UMAP
coords_base = subset_df[["umap_base_x", "umap_base_y"]].values

# Only calculate if we have at least 2 clusters and enough samples
rhythm_labels = subset_df["rhythm_cluster"].unique()
other_labels = subset_df["other_cluster"].unique()

print("\nSilhouette Score Analysis (Baseline UMAP):")
if len(rhythm_labels) >= 2 and len(subset_df) > len(rhythm_labels):
    sil_rhythm = silhouette_score(coords_base, subset_df["rhythm_cluster"])
    print(f"Silhouette for Rhythm-based clustering: {sil_rhythm:.4f}")
else:
    print("Not enough distinct Rhythm labels or samples for silhouette score")

if len(other_labels) >= 2 and len(subset_df) > len(other_labels):
    sil_other = silhouette_score(coords_base, subset_df["other_cluster"])
    print(f"Silhouette for 'Other' label-based clustering: {sil_other:.4f}")
else:
    print("Not enough distinct 'Other' labels or samples for silhouette score")

# Calculate silhouette scores for fine-tuned UMAP
coords_ft = subset_df[["umap_ft_x", "umap_ft_y"]].values

print("\nSilhouette Score Analysis (Fine-tuned UMAP):")
if len(rhythm_labels) >= 2 and len(subset_df) > len(rhythm_labels):
    sil_rhythm_ft = silhouette_score(coords_ft, subset_df["rhythm_cluster"])
    print(f"Silhouette for Rhythm-based clustering: {sil_rhythm_ft:.4f}")
else:
    print("Not enough distinct Rhythm labels or samples for silhouette score")

if len(other_labels) >= 2 and len(subset_df) > len(other_labels):
    sil_other_ft = silhouette_score(coords_ft, subset_df["other_cluster"])
    print(f"Silhouette for 'Other' label-based clustering: {sil_other_ft:.4f}")
else:
    print("Not enough distinct 'Other' labels or samples for silhouette score")

# %% [markdown]
# ## Visualization: Rhythm-Driven Clustering

# %%
# Create a side-by-side visualization showing how dual-labeled records cluster with their Rhythm counterparts
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

def plot_embedding(ax, emb_2d, df_single, df_dual, title):
    
    # (1) Plot all points in light gray
    ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        color="lightgray",
        edgecolor="none",
        s=40,
        alpha=0.4,
        label="All Records",
    )
    
    # (2) Plot single-labeled Rhythm records with their respective colors
    for rhythm_label in LABELS_OF_INTEREST:
        subset_mask = df_single["rhythm_label"] == rhythm_label
        sdf = df_single[subset_mask]
        if len(sdf) == 0:
            continue
            
        for _, row in sdf.iterrows():
            row_idx = row["row_idx"]
            ax.scatter(
                emb_2d[row_idx, 0],
                emb_2d[row_idx, 1],
                color=FIXED_LABEL_COLORS.get(rhythm_label, "gray"),
                marker="o",
                s=80,
                alpha=0.6,
                edgecolors="white",
                linewidth=0.5,
            )
    
    # (3) Plot dual-labeled Rhythm records with their respective colors but different markers
    for rhythm_label in LABELS_OF_INTEREST:
        subset_mask = df_dual["rhythm_label"] == rhythm_label
        ddf = df_dual[subset_mask]
        if len(ddf) == 0:
            continue
            
        for _, row in ddf.iterrows():
            row_idx = row["row_idx"]
            ax.scatter(
                emb_2d[row_idx, 0],
                emb_2d[row_idx, 1],
                color=FIXED_LABEL_COLORS.get(rhythm_label, "gray"),
                marker="^",  # Triangle marker for dual-labeled
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=f"Multi: {rhythm_label}+ (n={len(ddf)})",
            )
    
    ax.set_xlabel("UMAP Dim 1", fontsize=12)
    ax.set_ylabel("UMAP Dim 2", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

# Plot for baseline and fine-tuned embeddings
plot_embedding(axes[0], baseline_umap, df_single_rhythm, df_dual_rhythm, "Baseline (Pre-trained) Model")
plot_embedding(axes[1], finetuned_umap, df_single_rhythm, df_dual_rhythm, "Fine-tuned (Chapman) Model")

# Build legend
handles = []

# Handle for "All Records"
handles.append(
    Line2D(
        [0],
        [0],
        marker="o",
        color="lightgray",
        label="All Records",
        markersize=10,
        markeredgecolor="none",
        linewidth=0,
    )
)

# Add Rhythm labels to legend
handles.append(Patch(color="none", label="\nRhythm Labels:"))
for rhythm_label in LABELS_OF_INTEREST:
    # Single-labeled
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FIXED_LABEL_COLORS.get(rhythm_label, "gray"),
            markersize=10,
            label=f"  Single-labeled: {rhythm_label}",
            linewidth=0,
        )
    )
    # Dual-labeled
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=FIXED_LABEL_COLORS.get(rhythm_label, "gray"),
            markeredgecolor="black",
            markersize=10,
            label=f"  Dual-labeled: {rhythm_label} + other",
            linewidth=0,
        )
    )

legend = fig.legend(
    handles=handles,
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    title="Rhythm Label\nClustering",
)

plt.tight_layout()
plt.savefig(
    "rhythm_clustering_visualization.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## Distance to Centroids Analysis

# %%
# Calculate centroids for each rhythm label
rhythm_centroids = {}
for rhythm_label in LABELS_OF_INTEREST:
    # Get all records with this rhythm label (single + dual)
    rhythm_mask = subset_df["rhythm_cluster"] == rhythm_label
    if sum(rhythm_mask) > 0:
        # Calculate centroid for baseline
        base_coords = subset_df.loc[rhythm_mask, ["umap_base_x", "umap_base_y"]].values
        rhythm_centroids[f"{rhythm_label}_base"] = np.mean(base_coords, axis=0)
        
        # Calculate centroid for fine-tuned
        ft_coords = subset_df.loc[rhythm_mask, ["umap_ft_x", "umap_ft_y"]].values
        rhythm_centroids[f"{rhythm_label}_ft"] = np.mean(ft_coords, axis=0)

# Calculate distances to centroids
for rhythm_label in LABELS_OF_INTEREST:
    # Skip if no centroid for this label
    if f"{rhythm_label}_base" not in rhythm_centroids:
        continue
        
    # Calculate distances for baseline
    base_centroid = rhythm_centroids[f"{rhythm_label}_base"]
    base_coords = subset_df[["umap_base_x", "umap_base_y"]].values
    subset_df[f"dist_to_{rhythm_label}_base"] = np.sqrt(
        np.sum((base_coords - base_centroid) ** 2, axis=1)
    )
    
    # Calculate distances for fine-tuned
    ft_centroid = rhythm_centroids[f"{rhythm_label}_ft"]
    ft_coords = subset_df[["umap_ft_x", "umap_ft_y"]].values
    subset_df[f"dist_to_{rhythm_label}_ft"] = np.sqrt(
        np.sum((ft_coords - ft_centroid) ** 2, axis=1)
    )

# Compare distances for single vs. dual-labeled records
print("\nDistance to Centroid Analysis:")
for rhythm_label in LABELS_OF_INTEREST:
    # Skip if no centroid for this label
    if f"{rhythm_label}_base" not in rhythm_centroids:
        print(f"Skipping {rhythm_label} - not enough data")
        continue
        
    # Get single-labeled records for this rhythm
    single_mask = subset_df["rhythm_label"] == rhythm_label
    single_mask &= subset_df["other_label"].isna()  # Single-labeled records have no other label
    
    # Get dual-labeled records for this rhythm
    dual_mask = subset_df["rhythm_label"] == rhythm_label
    dual_mask &= ~subset_df["other_label"].isna()  # Dual-labeled records have an other label
    
    if sum(single_mask) == 0 or sum(dual_mask) == 0:
        print(f"Skipping {rhythm_label} - not enough single or dual-labeled data")
        continue
    
    # Baseline distances
    single_dist_base = subset_df.loc[single_mask, f"dist_to_{rhythm_label}_base"].mean()
    dual_dist_base = subset_df.loc[dual_mask, f"dist_to_{rhythm_label}_base"].mean()
    
    # Fine-tuned distances
    single_dist_ft = subset_df.loc[single_mask, f"dist_to_{rhythm_label}_ft"].mean()
    dual_dist_ft = subset_df.loc[dual_mask, f"dist_to_{rhythm_label}_ft"].mean()
    
    print(f"\n{rhythm_label}:")
    print(f"  Baseline - Single: {single_dist_base:.4f}, Multi: {dual_dist_base:.4f}, Ratio: {dual_dist_base/single_dist_base:.2f}x")
    print(f"  Fine-tuned - Single: {single_dist_ft:.4f}, Multi: {dual_dist_ft:.4f}, Ratio: {dual_dist_ft/single_dist_ft:.2f}x")

# %% [markdown]
# ## Faceted Visualization by Rhythm Label

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_faceted_plots(df_single, df_dual, df_no_interest, umap_data, title, figsize=(15, 12)):
    """
    Create faceted plots for each rhythm label, showing single-labeled records
    and dual-labeled records with the same rhythm label.
    
    Parameters:
    -----------
    df_single : DataFrame
        DataFrame containing single-labeled records
    df_dual : DataFrame
        DataFrame containing dual-labeled records
    df_no_interest : DataFrame
        DataFrame containing records with no label of interest
    umap_data : ndarray
        UMAP embeddings for all records
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    display_labels = LABELS_OF_INTEREST
    
    n_labels = len(display_labels)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, lbl in enumerate(display_labels):
        if i >= len(axes):
            break

        ax = axes[i]
        
        if lbl == "Other":
            continue
        else:
            # Regular case for rhythm labels
            # Get single-labeled records for this rhythm
            single_lbl = df_single[df_single["rhythm_label"] == lbl].copy()
            
            # Get dual-labeled records for this rhythm
            dual_lbl = df_dual[df_dual["rhythm_label"] == lbl].copy()

            if len(single_lbl) < 5 and len(dual_lbl) < 2:
                ax.text(
                    0.5,
                    0.5,
                    f"Not enough data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot background of all points in light gray
            ax.scatter(
                umap_data[:, 0],
                umap_data[:, 1],
                color="lightgray",
                edgecolor="none",
                s=20,
                alpha=0.2,
                label="All Records",
            )

            # Plot single-labeled records
            if len(single_lbl) > 0:
                single_points = np.array([umap_data[idx] for idx in single_lbl["row_idx"]])
                
                # Plot single-labeled points
                ax.scatter(
                    single_points[:, 0],
                    single_points[:, 1],
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                    marker="o",
                    s=80,
                    alpha=0.6,
                    edgecolors="white",
                    linewidth=0.5,
                    label=f"Single: {lbl} (n={len(single_lbl)})",
                )

            # Plot dual-labeled records
            if len(dual_lbl) > 0:
                dual_points = np.array([umap_data[idx] for idx in dual_lbl["row_idx"]])
                
                # Plot dual-labeled points
                ax.scatter(
                    dual_points[:, 0],
                    dual_points[:, 1],
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                    marker="^",
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Multi: {lbl}+ (n={len(dual_lbl)})",
                )
                
            ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
            ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
            
            # Create divider for marginal plots
            divider = make_axes_locatable(ax)

            # Add top marginal plot (x-dimension)
            ax_top = divider.append_axes("top", size="15%", pad=0.1)
            ax_top.tick_params(
                axis="both", which="both", labelbottom=False, labelleft=False
            )
            
            # Set the x-limits to match the main plot
            ax_top.set_xlim(ax.get_xlim())
            
            # Add KDE to top marginal plot
            if len(single_lbl) >= 5:
                try:
                    # For PACE, which has few samples, use a more robust approach
                    if lbl == "PACE":
                        bw_adjust = 0.5  # Use larger bandwidth for smoother KDE
                        sns.kdeplot(
                            x=single_points[:, 0], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            ax=ax_top,
                            label="Single",
                            bw_adjust=bw_adjust,
                            common_norm=False
                        )
                    else:
                        sns.kdeplot(
                            x=single_points[:, 0], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            ax=ax_top,
                            label="Single"
                        )
                except Exception:
                    pass
                    
            # Use a higher threshold for dual-labeled records to ensure enough data points
            min_dual_points = 5 if lbl != "PACE" else 2
            if len(dual_lbl) >= min_dual_points:
                try:
                    # For PACE, which has few samples, use a more robust approach
                    if lbl == "PACE":
                        bw_adjust = 0.5  # Use larger bandwidth for smoother KDE
                        sns.kdeplot(
                            x=dual_points[:, 0], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            linestyle="--",
                            ax=ax_top,
                            label="Dual",
                            bw_adjust=bw_adjust,
                            common_norm=False
                        )
                    else:
                        sns.kdeplot(
                            x=dual_points[:, 0], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            linestyle="--",
                            ax=ax_top,
                            label="Dual"
                        )
                except Exception:
                    pass
                
            ax_top.set_title(f"Label: {lbl}", fontsize=14, pad=5)
            ax_top.set_ylabel("")
            ax_top.set_yticks([])
            
            # Add a small legend to the KDE plot to explain line styles
            if len(single_lbl) >= 5 and len(dual_lbl) >= min_dual_points:
                ax_top.legend(["Single", "Dual"], fontsize=8)

            # Add right marginal plot (y-dimension)
            ax_right = divider.append_axes("right", size="15%", pad=0.1)
            ax_right.tick_params(
                axis="both", which="both", labelbottom=False, labelleft=False
            )
            
            # Set the y-limits to match the main plot
            ax_right.set_ylim(ax.get_ylim())
            
            # Add KDE to right marginal plot
            if len(single_lbl) >= 5:
                try:
                    # For PACE, which has few samples, use a more robust approach
                    if lbl == "PACE":
                        bw_adjust = 0.5  # Use larger bandwidth for smoother KDE
                        sns.kdeplot(
                            y=single_points[:, 1], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            ax=ax_right,
                            fill=False,
                            bw_adjust=bw_adjust,
                            common_norm=False
                        )
                    else:
                        sns.kdeplot(
                            y=single_points[:, 1], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            ax=ax_right
                        )
                except Exception:
                    pass
                    
            # Use a higher threshold for dual-labeled records to ensure enough data points
            min_dual_points = 5 if lbl != "PACE" else 2
            if len(dual_lbl) >= min_dual_points:
                try:
                    # For PACE, which has few samples, use a more robust approach
                    if lbl == "PACE":
                        bw_adjust = 0.5  # Use larger bandwidth for smoother KDE
                        sns.kdeplot(
                            y=dual_points[:, 1], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            linestyle="--",
                            ax=ax_right,
                            fill=False,
                            bw_adjust=bw_adjust,
                            common_norm=False
                        )
                    else:
                        sns.kdeplot(
                            y=dual_points[:, 1], 
                            color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                            linestyle="--",
                            ax=ax_right
                        )
                except Exception:
                    pass
                
            ax_right.set_xlabel("")
            ax_right.set_xticks([])
            
            ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
            ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
            
            # Add legend to each facet with counts
            ax.legend(loc="upper left", fontsize=8)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

# Create faceted plots for baseline and fine-tuned embeddings
baseline_faceted = create_faceted_plots(
    df_single_rhythm, 
    df_dual_rhythm,
    df_no_rhythm,
    baseline_umap, 
    "Baseline (Pre-trained) Model - Faceted by Rhythm Label"
)
plt.savefig("results/baseline_faceted_plots.png", dpi=150, bbox_inches="tight")
plt.show()

finetuned_faceted = create_faceted_plots(
    df_single_rhythm, 
    df_dual_rhythm,
    df_no_rhythm,
    finetuned_umap, 
    "Fine-tuned (Chapman) Model - Faceted by Rhythm Label"
)
plt.savefig("results/finetuned_faceted_plots.png", dpi=150, bbox_inches="tight")
plt.show()

#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def create_metadata_faceted_plots(
    df_records, 
    umap_data, 
    metadata_col,
    title="Faceted Plots by Metadata",
    continuous=False,
    figsize=(15, 12),
    cmap_name="viridis"
):
    """
    Create facet plots for each rhythm label, colored by metadata.
    
    Parameters:
    -----------
    df_records : DataFrame
        DataFrame containing record information with metadata
    umap_data : ndarray
        UMAP embeddings for all records
    metadata_col : str
        Which metadata field to color by (e.g., 'age', 'is_male')
    title : str
        Title for the plot
    continuous : bool
        If True, treat metadata_col as numeric and use a continuous color scale
        If False, treat it as categorical
    figsize : tuple
        Figure size
    cmap_name : str
        Matplotlib colormap name if continuous = True
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Filter records with valid metadata
    df_valid = df_records[df_records[metadata_col].notna()].copy()
    
    if len(df_valid) == 0:
        print(f"No records with valid {metadata_col} data")
        return None
    
    # Add rhythm_label to the dataframe
    df_valid["rhythm_label"] = df_valid.apply(get_rhythm_label, axis=1)
    
    # Filter out records with no rhythm label
    df_valid = df_valid[df_valid["rhythm_label"].notna()]
    
    if len(df_valid) == 0:
        print("No records with valid rhythm labels")
        return None
    
    # Prepare display labels
    display_labels = LABELS_OF_INTEREST
    
    n_labels = len(display_labels)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    # For continuous data, set up color normalization
    if continuous:
        valid_meta = df_valid[metadata_col].dropna()
        
        if metadata_col.lower() == 'age':
            # Clip age to reasonable range
            df_valid[metadata_col] = df_valid[metadata_col].clip(lower=0, upper=90)
            valid_meta = df_valid[metadata_col].dropna()
            # For age, explicitly set the range to 0-90
            vmin = 0
            vmax = 90
        else:
            if len(valid_meta) > 0:
                vmin = valid_meta.min()
                vmax = valid_meta.max()
            else:
                vmin, vmax = 0, 1  # fallback
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
    else:
        # For categorical data, create a color mapping
        unique_vals = df_valid[metadata_col].dropna().unique()
        unique_vals = sorted(unique_vals, key=lambda x: str(x))
        palette = sns.color_palette("tab10", n_colors=len(unique_vals))
        cat2color = dict(zip(unique_vals, palette))
    
    for i, lbl in enumerate(display_labels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter records for this rhythm label
        df_rhythm = df_valid[df_valid["rhythm_label"] == lbl].copy()
        
        if len(df_rhythm) < 5:
            ax.text(
                0.5,
                0.5,
                f"Not enough data for {lbl}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Label: {lbl}", fontsize=14)
            continue
        
        # Get UMAP coordinates for this rhythm
        rhythm_points = np.array([umap_data[idx] for idx in df_rhythm["row_idx"]])
        
        # Create a DataFrame for the points for easier KDE plotting
        df_rhythm_points = pd.DataFrame({
            'x': rhythm_points[:, 0],
            'y': rhythm_points[:, 1],
            'metadata': df_rhythm[metadata_col].values
        })
        
        # Plot background of all points in light gray
        ax.scatter(
            umap_data[:, 0],
            umap_data[:, 1],
            color="lightgray",
            edgecolor="none",
            s=20,
            alpha=0.2,
            label="All Records",
        )

        # Now plot points colored by metadata
        if continuous:
            # For numeric/continuous data
            cvals = []
            for idx, row in df_rhythm.iterrows():
                val = row[metadata_col]
                if pd.isna(val):
                    cvals.append("black")
                else:
                    cvals.append(cmap(norm(val)))
                    
            scatter = ax.scatter(
                rhythm_points[:, 0],
                rhythm_points[:, 1],
                c=cvals[:len(rhythm_points)],
                s=80,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )
        else:
            # For categorical data
            for cat in unique_vals:
                cat_indices = df_rhythm[df_rhythm[metadata_col] == cat].index
                cat_points = np.array([rhythm_points[i] for i, idx in enumerate(df_rhythm.index) if idx in cat_indices and i < len(rhythm_points)])
                
                if len(cat_points) > 0:
                    label_text = "Male" if cat and metadata_col == "is_male" else "Female" if not cat and metadata_col == "is_male" else f"{cat}"
                    ax.scatter(
                        cat_points[:, 0],
                        cat_points[:, 1],
                        color=cat2color[cat],
                        s=80,
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.5,
                        label=label_text,
                    )
        
        # Create divider for marginal plots
        divider = make_axes_locatable(ax)
        
        # Add top marginal plot (x-dimension)
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Set the x-limits to match the main plot
        ax_top.set_xlim(ax.get_xlim())
        
        # Add KDE to top marginal plot
        if len(df_rhythm_points) >= 5:
            if continuous:
                # For continuous data, show a single KDE
                try:
                    sns.kdeplot(
                        x=df_rhythm_points['x'],
                        color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                        ax=ax_top,
                        fill=True,
                        alpha=0.3,
                    )
                except Exception as e:
                    print(f"KDE error for {lbl} (top): {e}")
                    pass
                    
            else:
                # For categorical data, show separate KDEs for each category
                for cat in unique_vals:
                    cat_data = df_rhythm_points[df_rhythm_points['metadata'] == cat]
                    if len(cat_data) >= 5:
                        try:
                            label_text = "Male" if cat and metadata_col == "is_male" else "Female" if not cat and metadata_col == "is_male" else f"{cat}"
                            sns.kdeplot(
                                x=cat_data['x'],
                                color=cat2color[cat],
                                ax=ax_top,
                                fill=True,
                                alpha=0.3,
                                label=label_text
                            )
                        except Exception as e:
                            print(f"KDE error for {lbl}, {cat} (top): {e}")
                            pass
                
        ax_top.set_title(f"Label: {lbl}", fontsize=14, pad=5)
        ax_top.set_ylabel("")
        ax_top.set_yticks([])

        # Add right marginal plot (y-dimension)
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Set the y-limits to match the main plot
        ax_right.set_ylim(ax.get_ylim())
        
        # Add KDE to right marginal plot
        if len(df_rhythm_points) >= 5:
            if continuous:
                # For continuous data, show a single KDE
                try:
                    sns.kdeplot(
                        y=df_rhythm_points['y'],
                        color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                        ax=ax_right,
                        fill=True,
                        alpha=0.3,
                    )
                except Exception as e:
                    print(f"KDE error for {lbl} (right): {e}")
                    pass
                    
            else:
                # For categorical data, show separate KDEs for each category
                for cat in unique_vals:
                    cat_data = df_rhythm_points[df_rhythm_points['metadata'] == cat]
                    if len(cat_data) >= 5:
                        try:
                            sns.kdeplot(
                                y=cat_data['y'],
                                color=cat2color[cat],
                                ax=ax_right,
                                fill=True,
                                alpha=0.3,
                            )
                        except Exception as e:
                            print(f"KDE error for {lbl}, {cat} (right): {e}")
                            pass
                
        ax_right.set_xlabel("")
        ax_right.set_xticks([])
        
        ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
        ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
        
        # Add legend to each facet
        if not continuous:
            ax.legend(loc="upper left", fontsize=8)
            # For categorical data, add a legend to the top marginal plot as well
            if len(unique_vals) <= 3:  # Only add legend if not too many categories
                ax_top.legend(fontsize=8)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add a colorbar for continuous data
    if continuous:
        fig.subplots_adjust(right=0.85)  # Adjust to make room for colorbar
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])  # x, y, width, height - moved more to the left
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(metadata_col)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    
    return fig

# Create faceted plots for age metadata
print("Creating faceted plots for age metadata...")
baseline_age_faceted = create_metadata_faceted_plots(
    df_records,
    baseline_umap,
    metadata_col="age",
    title="Baseline Model - Faceted by Rhythm Label, Colored by Age",
    continuous=True,
    cmap_name="plasma",
)
if baseline_age_faceted:
    plt.savefig("results/baseline_age_faceted_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

finetuned_age_faceted = create_metadata_faceted_plots(
    df_records,
    finetuned_umap,
    metadata_col="age",
    title="Fine-tuned Model - Faceted by Rhythm Label, Colored by Age",
    continuous=True,
    cmap_name="plasma",
)
if finetuned_age_faceted:
    plt.savefig("results/finetuned_age_faceted_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

# Create faceted plots for sex metadata
print("Creating faceted plots for sex metadata...")
baseline_sex_faceted = create_metadata_faceted_plots(
    df_records,
    baseline_umap,
    metadata_col="is_male",
    title="Baseline Model - Faceted by Rhythm Label, Colored by Sex",
    continuous=False,
)
if baseline_sex_faceted:
    plt.savefig("results/baseline_sex_faceted_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

finetuned_sex_faceted = create_metadata_faceted_plots(
    df_records,
    finetuned_umap,
    metadata_col="is_male",
    title="Fine-tuned Model - Faceted by Rhythm Label, Colored by Sex",
    continuous=False,
)
if finetuned_sex_faceted:
    plt.savefig("results/finetuned_sex_faceted_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#%%
def create_inverted_dual_label_faceted_plots(
    df_dual, 
    df_all, 
    umap_data, 
    title, 
    figsize=(15, 12)
):
    """
    Create faceted plots for each dual label, where each facet shows a different rhythm label.
    Uses a fixed grid layout with all rhythm labels, showing empty facets with text when no data exists.
    
    Parameters:
    -----------
    df_dual : DataFrame
        DataFrame containing dual-labeled records
    df_all : DataFrame
        DataFrame containing all records
    umap_data : ndarray
        UMAP embeddings for the model
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Get all unique other_labels that have dual-labeled records
    all_other_labels = df_dual["other_label"].dropna().unique().tolist()
    
    # Use all rhythm labels of interest for consistent facet layout
    display_labels = LABELS_OF_INTEREST
    
    # Create a figure for each other_label
    for other_label in all_other_labels:
        # Get all dual-labeled records for this other_label
        dual_records = df_dual[df_dual["other_label"] == other_label].copy()
        
        if len(dual_records) < 2:
            print(f"Not enough dual-labeled records for {other_label}. Skipping.")
            continue
            
        # Set up the grid for facets - always use a fixed layout
        n_labels = len(display_labels)
        n_cols = 3
        n_rows = (n_labels + n_cols - 1) // n_cols
        
        # Create figure with consistent size
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(f"{title} - Multi Label: {other_label}", fontsize=16, y=0.98)
        
        # Flatten axes array for easier indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        # Create a facet for each rhythm_label (fixed layout)
        for i, rhythm_label in enumerate(display_labels):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get dual-labeled records with this specific combination
            dual_subset = dual_records[dual_records["rhythm_label"] == rhythm_label].copy()
            
            # Plot ALL points in light gray (background)
            ax.scatter(
                umap_data[:, 0],
                umap_data[:, 1],
                color="lightgray",
                edgecolor="none",
                s=20,
                alpha=0.2,
            )
            
            if len(dual_subset) < 2:
                # Show empty facet with text
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {rhythm_label}+{other_label}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12
                )
                ax.set_title(f"Label: {rhythm_label}", fontsize=14)
            else:
                # Plot dual-labeled points
                dual_points = np.array([umap_data[idx] for idx in dual_subset["row_idx"]])
                ax.scatter(
                    dual_points[:, 0],
                    dual_points[:, 1],
                    color=FIXED_LABEL_COLORS.get(rhythm_label, "gray"),
                    marker="^",
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Multi: {rhythm_label}+ (n={len(dual_subset)})",
                )
                
                # Create divider for marginal plots
                divider = make_axes_locatable(ax)
                
                # Add top marginal plot (x-dimension)
                ax_top = divider.append_axes("top", size="15%", pad=0.1)
                ax_top.tick_params(
                    axis="both", which="both", labelbottom=False, labelleft=False
                )
                
                # Set the x-range to match the main plot
                x_min, x_max = ax.get_xlim()
                ax_top.set_xlim(x_min, x_max)
                
                # Add KDE to top marginal plot
                if len(dual_points) >= 2:
                    try:
                        # Special handling for small sample sizes
                        bw_adjust = 0.5 if len(dual_points) < 10 else 1.0
                        sns.kdeplot(
                            x=dual_points[:, 0], 
                            color=FIXED_LABEL_COLORS.get(rhythm_label, "gray"), 
                            ax=ax_top,
                            fill=True,
                            alpha=0.3,
                            bw_adjust=bw_adjust
                        )
                    except Exception as e:
                        print(f"KDE error for {rhythm_label}+{other_label} (top): {e}")
                
                ax_top.set_ylabel("")
                ax_top.set_yticks([])
                
                # Add right marginal plot (y-dimension)
                ax_right = divider.append_axes("right", size="15%", pad=0.1)
                ax_right.tick_params(
                    axis="both", which="both", labelbottom=False, labelleft=False
                )
                
                # Set the y-range to match the main plot
                y_min, y_max = ax.get_ylim()
                ax_right.set_ylim(y_min, y_max)
                
                # Add KDE to right marginal plot
                if len(dual_points) >= 2:
                    try:
                        # Special handling for small sample sizes
                        bw_adjust = 0.5 if len(dual_points) < 10 else 1.0
                        sns.kdeplot(
                            y=dual_points[:, 1], 
                            color=FIXED_LABEL_COLORS.get(rhythm_label, "gray"), 
                            ax=ax_right,
                            fill=True,
                            alpha=0.3,
                            bw_adjust=bw_adjust
                        )
                    except Exception as e:
                        print(f"KDE error for {rhythm_label}+{other_label} (right): {e}")
                
                ax_right.set_xlabel("")
                ax_right.set_xticks([])
                
                # Add title to the top marginal plot
                ax_top.set_title(f"Label: {rhythm_label}", fontsize=14, pad=5)
                
                # Add legend to each facet with counts
                ax.legend(loc="upper left", fontsize=8)
            
            # Set axis labels - only show y-labels for leftmost plots and x-labels for bottom plots
            ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
            ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{other_label}_{title.replace(' ', '_')}_faceted_plots.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    return

# Update the code to call the function separately for baseline and fine-tuned
print("Creating inverted dual-label faceted plots for baseline model...")
create_inverted_dual_label_faceted_plots(
    df_dual_rhythm,
    df_single_rhythm,
    baseline_umap,
    "Baseline (Pre-trained) Model"
)

print("Creating inverted dual-label faceted plots for fine-tuned model...")
create_inverted_dual_label_faceted_plots(
    df_dual_rhythm,
    df_single_rhythm,
    finetuned_umap,
    "Fine-tuned (Chapman) Model"
)
