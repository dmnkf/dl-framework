#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Embedding Space Analysis (Pre-trained vs Fine-tuned)
-------------------------------------------------------------
UMAP on the FULL test set (single + multi-labeled),
highlighting single-labeled records in color.
"""

# %% [markdown]
# # Phase 1 Notebook
#
# In this notebook-style script, we compare **Baseline (pre-trained)** vs. **Fine-tuned** embeddings on the full test set.
# We highlight **single-labeled** records in color while the rest are shown in gray.

# %% [markdown]
# ## Imports
# Here, we import Python standard libraries and our local project modules.

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
import matplotlib.patheffects as PathEffects

# Ensure project path is in sys.path
project_root = Path().absolute().parent.parent
sys.path.append(str(project_root))

# Our project modules
from src.visualization.embedding_viz import run_umap
from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

# Matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 12


# %% [markdown]
# ## Color Palettes and Helpers
# We define fixed colors for specific labels, as well as group-based color assignments.

# %%
SINGLE_COLOR_PALETTE = sns.color_palette("colorblind", 5)

FIXED_LABEL_COLORS = {
    "SR": SINGLE_COLOR_PALETTE[0], 
    "SB": SINGLE_COLOR_PALETTE[1],  
    "AFIB": SINGLE_COLOR_PALETTE[2],
    "GSVT": SINGLE_COLOR_PALETTE[3], 
    "PACE": SINGLE_COLOR_PALETTE[4],  
}

FIXED_LABEL_MARKERS = {
    "SR": ".", 
    "AFIB": "s",
    "SB": "*",
    "GSVT": "D",
    "PACE": "X",
}

COLOR_PALETTES = {
    "Rhythm": SINGLE_COLOR_PALETTE,
    "Morphology": SINGLE_COLOR_PALETTE,
    "Duration": SINGLE_COLOR_PALETTE,
    "Amplitude": SINGLE_COLOR_PALETTE,
    "Other": SINGLE_COLOR_PALETTE,
}


def get_group_color_map(df_labels):
    """
    Generate a dict: group_label_map[group][label] -> color.
    df_labels must have 'integration_name' and 'group'.
    Only uses colors from FIXED_LABEL_COLORS.
    """
    # Get all unique labels
    all_labels = df_labels["integration_name"].unique()

    # Map each unique label to a color
    label_to_color = {}

    # First, use fixed colors for specific labels
    for label in all_labels:
        if label in FIXED_LABEL_COLORS:
            label_to_color[label] = FIXED_LABEL_COLORS[label]

    # For remaining labels, assign colors from FIXED_LABEL_COLORS in a round-robin fashion
    available_colors = list(FIXED_LABEL_COLORS.values())
    color_idx = 0
    
    for label in all_labels:
        if label not in label_to_color:
            # Assign the next available color from FIXED_LABEL_COLORS
            label_to_color[label] = available_colors[color_idx % len(available_colors)]
            color_idx += 1

    # Create the group structure
    group_label_map = {}
    for _, row in df_labels.iterrows():
        label = row["integration_name"]
        group = row["group"]
        if group not in group_label_map:
            group_label_map[group] = {}
        if label not in group_label_map[group]:
            group_label_map[group][label] = label_to_color[label]

    return group_label_map


# Distinct markers for each group
GROUP_MARKERS = {
    "Rhythm": "o",
    "Morphology": "s",
    "Duration": "^",
    "Amplitude": "D",
    "Other": "X",
}

# %% [markdown]
# ## Data Loading
# Here, we load the **Arrhythmia (Chapman) test set** from our `UnifiedDataset`.
# We then retrieve embeddings for **Baseline** and **Fine-Tuned** models.

# %%
print("Phase 1: UMAP on FULL data, highlighting single-labeled records.")
print("=" * 70)

arr_data = UnifiedDataset(
    Path(project_root) / "data", modality=DatasetModality.ECG, dataset_key="arrhythmia"
)
arr_splits = arr_data.get_splits()
arr_test_ids = arr_splits.get("test", [])

arr_md_store = arr_data.metadata_store

pretrained_embedding = "baseline"
finetuned_embedding = "fine_tuned_50"

records_info = []
emb_base_list = []
emb_ft_list = []

for rid in arr_test_ids:
    meta = arr_md_store.get(rid, {})
    labels_meta = meta.get("labels_metadata", [])

    try:
        emb_base = arr_data.get_embeddings(rid, embeddings_type=pretrained_embedding)
        emb_ft = arr_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
    except Exception as e:
        print(f"Skipping {rid} (missing embeddings). Err: {e}")
        continue

    records_info.append(
        {"record_id": rid, "labels_meta": labels_meta, "n_labels": len(labels_meta)}
    )
    emb_base_list.append(emb_base)
    emb_ft_list.append(emb_ft)

if not records_info:
    print("No records found. Exiting.")
    sys.exit()

df_records = pd.DataFrame(records_info)
df_records["row_idx"] = df_records.index  # 0..N-1

# Stack embeddings
baseline_embeddings = np.vstack(emb_base_list)
finetuned_embeddings = np.vstack(emb_ft_list)

print(f"Total records loaded: {len(df_records)}")
print(" - Baseline shape:", baseline_embeddings.shape)
print(" - Fine-tuned shape:", finetuned_embeddings.shape)


# %% [markdown]
# ## Single-Labeled Subset
# We now identify records that have **exactly one** label and store this subset separately.

# %%
mask_single = df_records["n_labels"] == 1
df_single = df_records[mask_single].copy()

df_single["integration_name"] = df_single["labels_meta"].apply(
    lambda lm: lm[0].get("integration_name", "unknown") if len(lm) == 1 else "unknown"
)
df_single["group"] = df_single["labels_meta"].apply(
    lambda lm: lm[0].get("group", "Other") if len(lm) == 1 else "Other"
)

print("Single-labeled records:", len(df_single))


# %% [markdown]
# ## UMAP on the Full Dataset
# We run UMAP on **all records** (both single- and multi-labeled) for a global view, then highlight single-labeled in the plot.

# %%
print("\nRunning UMAP (baseline & fine-tuned) on all records...")

umap_params = dict(n_neighbors=30, min_dist=0.25, n_components=2, metric="euclidean", random_state=42)
baseline_umap = run_umap(baseline_embeddings, **umap_params)
finetuned_umap = run_umap(finetuned_embeddings, **umap_params)

print("UMAP finished.\n")

# Prepare color mapping for single-labeled points
single_labels = df_single[["integration_name", "group"]].drop_duplicates()
group_color_mapping = get_group_color_map(single_labels)


# %% [markdown]
# ## Visualization
# We create a **two-panel** figure comparing the **Baseline** vs. **Fine-tuned** spaces,
# using gray for all records and colorful markers for single-labeled examples.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# fig.suptitle(
#     "Chapman ECG Embedding Visualization: Baseline vs. Fine-Tuned Model",
#     fontsize=18,
#     fontweight="bold",
# )

def plot_embedding(ax, emb_2d, title):
    ax.set_title(title, fontsize=14)

    # (1) All points in light gray
    ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        color="lightgray",
        edgecolor="none",
        s=40,
        alpha=0.4,
        label="All Records",
    )

    # (2) Overlay only points with labels in FIXED_LABEL_COLORS
    for row in df_single.itertuples():
        row_idx = row.row_idx
        label_name = getattr(row, "integration_name")
        
        # Skip labels not in FIXED_LABEL_COLORS
        if label_name not in FIXED_LABEL_COLORS:
            continue
            
        color = FIXED_LABEL_COLORS[label_name]
        marker = FIXED_LABEL_MARKERS.get(label_name, "o")

        ax.scatter(
            emb_2d[row_idx, 0],
            emb_2d[row_idx, 1],
            c=[color],
            marker=marker,
            s=80,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("UMAP Dim 1", fontsize=12)
    ax.set_ylabel("UMAP Dim 2", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)


plot_embedding(axes[0], baseline_umap, "Baseline (Pre-trained) Model")
plot_embedding(axes[1], finetuned_umap, "Fine-tuned (Chapman) Model")

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

# Group fixed labels by their groups
fixed_labels_by_group = {}
for label in FIXED_LABEL_COLORS:
    # Find the group for this label
    label_group = None
    for row in df_single.itertuples():
        if getattr(row, "integration_name") == label:
            label_group = getattr(row, "group")
            break
    
    if label_group:
        if label_group not in fixed_labels_by_group:
            fixed_labels_by_group[label_group] = []
        fixed_labels_by_group[label_group].append(label)

# Add group headers and fixed labels to legend
for grp in fixed_labels_by_group:
    handles.append(Patch(color="none", label=f"\n{grp} Group:"))
    for lbl in fixed_labels_by_group[grp]:
        c = FIXED_LABEL_COLORS[lbl]
        mkr = FIXED_LABEL_MARKERS.get(lbl, "o")
        handles.append(
            Line2D(
                [0],
                [0],
                marker=mkr,
                color=c,
                markerfacecolor=c,
                markersize=10,
                label=f"  {lbl}",
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
    title="Single-Labeled\nGroups & Labels",
)

plt.savefig(
    "results/baseline_vs_finetuned_embedding_visualization.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## Additional Fine-tuned Models Visualization
# Now we load and visualize additional fine-tuned models (100 and 200 epochs)

# %%
print("\nLoading additional fine-tuned models (100 and 200 epochs)...")
finetuned_embedding_100 = "fine_tuned_100"
finetuned_embedding_200 = "fine_tuned_200"

emb_ft_100_list = []
emb_ft_200_list = []

for rid in arr_test_ids:
    try:
        emb_ft_100 = arr_data.get_embeddings(rid, embeddings_type=finetuned_embedding_100)
        emb_ft_200 = arr_data.get_embeddings(rid, embeddings_type=finetuned_embedding_200)
        emb_ft_100_list.append(emb_ft_100)
        emb_ft_200_list.append(emb_ft_200)
    except Exception as e:
        print(f"Skipping {rid} (missing embeddings for additional models). Err: {e}")

# Stack embeddings
finetuned_100_embeddings = np.vstack(emb_ft_100_list)
finetuned_200_embeddings = np.vstack(emb_ft_200_list)

print(" - Fine-tuned 100 shape:", finetuned_100_embeddings.shape)
print(" - Fine-tuned 200 shape:", finetuned_200_embeddings.shape)

# Run UMAP on additional models
print("\nRunning UMAP on additional fine-tuned models...")
finetuned_100_umap = run_umap(finetuned_100_embeddings, **umap_params)
finetuned_200_umap = run_umap(finetuned_200_embeddings, **umap_params)
print("UMAP finished for additional models.\n")

#%%
# Create a new figure for all four embedding spaces
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 16))
axes2 = axes2.flatten()  # Flatten to make indexing easier

# Plot all four embedding spaces
plot_embedding(axes2[0], baseline_umap, "Baseline (Pre-trained) Model")
plot_embedding(axes2[1], finetuned_umap, "Fine-tuned (Chapman) Model - 50 Epochs")
plot_embedding(axes2[2], finetuned_100_umap, "Fine-tuned (Chapman) Model - 100 Epochs")
plot_embedding(axes2[3], finetuned_200_umap, "Fine-tuned (Chapman) Model - 200 Epochs")

# Build legend for the second figure
handles2 = []

# Handle for "All Records"
handles2.append(
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

# Group fixed labels by their groups
fixed_labels_by_group = {}
for label in FIXED_LABEL_COLORS:
    # Find the group for this label
    label_group = None
    for row in df_single.itertuples():
        if getattr(row, "integration_name") == label:
            label_group = getattr(row, "group")
            break
    
    if label_group:
        if label_group not in fixed_labels_by_group:
            fixed_labels_by_group[label_group] = []
        fixed_labels_by_group[label_group].append(label)

# Add group headers and fixed labels to legend
for grp in fixed_labels_by_group:
    handles2.append(Patch(color="none", label=f"\n{grp} Group:"))
    for lbl in fixed_labels_by_group[grp]:
        c = FIXED_LABEL_COLORS[lbl]
        mkr = FIXED_LABEL_MARKERS.get(lbl, "o")
        handles2.append(
            Line2D(
                [0],
                [0],
                marker=mkr,
                color=c,
                markerfacecolor=c,
                markersize=10,
                label=f"  {lbl}",
                linewidth=0,
            )
        )

legend2 = fig2.legend(
    handles=handles2,
    loc="center right",
    bbox_to_anchor=(1.15, 0.5),
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    title="Single-Labeled\nGroups & Labels",
)

plt.tight_layout()
plt.savefig(
    "results/all_finetuned_embedding_visualization.png", dpi=150, bbox_inches="tight"
)
plt.show()
