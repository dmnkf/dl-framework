#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2.2: Subcluster Analysis within Single-Label Groups for PTBXL Dataset
---------------------------------------------------------------------------
UMAP on the FULL test set (single + multi-labeled),
focusing on records with two labels where one is a label of interest.
This helps identify potential subclustering within single-label groups.
"""

# %% [markdown]
# # Phase 2.2 - Subcluster Analysis Notebook for PTBXL Dataset
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
# Define the color palette exactly as in phase1.py and phase2.py
SINGLE_COLOR_PALETTE = sns.color_palette("Set1", 9)

# Fixed colors for specific labels - exactly matching phase1.py and phase2.py
FIXED_LABEL_COLORS = {
    "SR": SINGLE_COLOR_PALETTE[0], 
    "AFIB": SINGLE_COLOR_PALETTE[1],
    "SB": SINGLE_COLOR_PALETTE[3],  
    "GSVT": SINGLE_COLOR_PALETTE[4], 
    "PACE": SINGLE_COLOR_PALETTE[7],  
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

# PTBXL mapping for arrhythmia labels
PTBXL_ARR_MAP = {
    "AFIB": "AFIB",
    "AFLT": "AFIB",
    "SR": "SR",
    "SARRH": "SR",
    "SBRAD": "SB",
    "PACE": "PACE",
    "STACH": "GSVT",
    "SVARR": "GSVT",
    "SVTAC": "GSVT",
    "PSVT": "GSVT",
}

LABELS_OF_INTEREST = ["AFIB", "GSVT", "SB", "SR", "PACE"]

# %% [markdown]
# ## Data Loading & Metadata Extraction

# %%
print("Phase 2.2: Analyzing subclustering within single-label groups for PTBXL dataset.")
ptbxl_data = UnifiedDataset(
    Path(project_root) / "data", modality=DatasetModality.ECG, dataset_key="ptbxl"
)
ptbxl_splits = ptbxl_data.get_splits()
ptbxl_test_ids = ptbxl_splits.get("test", [])
ptbxl_md_store = ptbxl_data.metadata_store

# Extract demographic metadata for all test records
demographic_data = {}
for rid in ptbxl_test_ids:
    meta = ptbxl_md_store.get(rid, {})
    age = meta.get("age", None)
    sex = meta.get("sex", None)
    demographic_data[rid] = {
        "age": age if isinstance(age, (int, float)) else None,
        "sex": sex,
    }
print(f"Extracted demographic data for {len(demographic_data)} records")

pretrained_embedding = "baseline"
finetuned_embedding = "fine_tuned_50"

print(ptbxl_md_store.get(ptbxl_test_ids[0]))

#%%

def extract_ptbxl_records_info():
    records_info = []
    emb_base_list = []
    emb_ft_list = []
    
    for rid in ptbxl_test_ids:
        meta = ptbxl_md_store.get(rid, {})
        scp_codes = meta.get("scp_codes", {})
        scp_statements = meta.get("scp_statements", {})

        valid_rhythm_codes = []
        for code_key in scp_codes:
            if code_key in PTBXL_ARR_MAP:
                code_info = scp_statements.get(code_key, {})
                if code_info.get("rhythm", 0.0) == 1.0:
                    valid_rhythm_codes.append(code_key)

        # Skip records with no valid rhythm codes
        if not valid_rhythm_codes:
            continue
            
        # Get embeddings for this record
        try:
            emb_base = ptbxl_data.get_embeddings(rid, embeddings_type=pretrained_embedding)
            emb_ft = ptbxl_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
        except Exception:
            continue
            
        # Store additional metadata
        age = meta.get("age", np.nan)
        sex = meta.get("sex", np.nan)
        site = meta.get("site", np.nan)
        device = meta.get("device", "NA")
        
        # For single-labeled records
        if len(valid_rhythm_codes) == 1:
            code_key = valid_rhythm_codes[0]
            mapped_label = PTBXL_ARR_MAP[code_key]
            
            records_info.append({
                "record_id": rid,
                "ptbxl_code": code_key,
                "mapped_label": mapped_label,
                "n_labels": 1,
                "is_single_labeled": True,
                "is_dual_labeled": False,
                "age": age,
                "sex": sex,
                "site": site,
                "device": device,
            })
            
            emb_base_list.append(emb_base)
            emb_ft_list.append(emb_ft)
            
        # For dual-labeled records
        elif len(valid_rhythm_codes) > 1:
            # Sort by priority (using the order in LABELS_OF_INTEREST)
            mapped_labels = [PTBXL_ARR_MAP[code] for code in valid_rhythm_codes]
            priority_labels = [lbl for lbl in LABELS_OF_INTEREST if lbl in mapped_labels]
            
            if priority_labels:
                primary_label = priority_labels[0]
                other_labels = [lbl for lbl in mapped_labels if lbl != primary_label]
                other_label = other_labels[0] if other_labels else None
                
                records_info.append({
                    "record_id": rid,
                    "ptbxl_code": ",".join(valid_rhythm_codes),
                    "mapped_label": primary_label,
                    "other_label": other_label,
                    "n_labels": len(valid_rhythm_codes),
                    "is_single_labeled": False,
                    "is_dual_labeled": True,
                    "age": age,
                    "sex": sex,
                    "site": site,
                    "device": device,
                })
                
                emb_base_list.append(emb_base)
                emb_ft_list.append(emb_ft)
    
    # Create DataFrames
    df_records = pd.DataFrame(records_info)
    df_records["row_idx"] = df_records.index
    
    # Convert embeddings to numpy arrays
    baseline_umap = np.array(emb_base_list)
    finetuned_umap = np.array(emb_ft_list)
    
    return df_records, baseline_umap, finetuned_umap


records_info, emb_base_list, emb_ft_list = extract_ptbxl_records_info()
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
print(f" - With sex data: {df_records['sex'].notna().sum()}")

# Process records for visualization
df_records = pd.DataFrame(records_info)
df_records["row_idx"] = df_records.index

# Convert embeddings to numpy arrays
baseline_umap = np.array(emb_base_list)
finetuned_umap = np.array(emb_ft_list)

# Set rhythm_label and other_label columns
df_records["rhythm_label"] = df_records["mapped_label"]

# Get single-labeled records with a label of interest
df_single_rhythm = df_records[
    (df_records["n_labels"] == 1)
    & (df_records["mapped_label"].isin(LABELS_OF_INTEREST))
].copy()
print(f"Single-labeled records with label of interest: {len(df_single_rhythm)}")

# Get records with no label of interest
df_no_interest = df_records[
    ~df_records["mapped_label"].isin(LABELS_OF_INTEREST)
].copy()
print(f"Records with no label of interest: {len(df_no_interest)}")

# Get dual-labeled records with at least one label of interest
df_dual_rhythm = df_records[
    (df_records["n_labels"] > 1) & 
    (df_records["mapped_label"].isin(LABELS_OF_INTEREST))
].copy()
print(f"Dual-labeled records with label of interest: {len(df_dual_rhythm)}")

# Print counts by rhythm label
print("\nSingle-labeled records by rhythm label:")
print(df_single_rhythm["rhythm_label"].value_counts())

print("\nDual-labeled records by primary rhythm label:")
print(df_dual_rhythm["rhythm_label"].value_counts())

print("\nDual-labeled records by secondary rhythm label:")
print(df_dual_rhythm["other_label"].value_counts())

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
# ## Plot UMAP Embeddings with Rhythm Labels

# %%
# Create a figure with two subplots for baseline and fine-tuned embeddings
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("UMAP Embeddings for PTBXL Dataset", fontsize=16)

def plot_embedding(ax, emb_2d, df_single, df_dual, title):
    """
    Plot UMAP embeddings with rhythm labels.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    emb_2d : ndarray
        UMAP embeddings
    df_single : DataFrame
        DataFrame containing single-labeled records
    df_dual : DataFrame
        DataFrame containing dual-labeled records
    title : str
        Title for the plot
    """
    # Plot single-labeled records
    for rhythm_label in LABELS_OF_INTEREST:
        if df_single.empty:
            continue
            
        mask = df_single["rhythm_label"] == rhythm_label
        if mask.sum() == 0:
            continue
        
        indices = df_single.loc[mask, "row_idx"].values
        color = FIXED_LABEL_COLORS.get(rhythm_label, "gray")
        
        ax.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            c=[color],
            marker="o",
            s=50,
            alpha=0.7,
            label=f"Single: {rhythm_label}"
        )
    
    # Plot dual-labeled records
    for rhythm_label in LABELS_OF_INTEREST:
        if df_dual.empty:
            continue
            
        mask = df_dual["rhythm_label"] == rhythm_label
        if mask.sum() == 0:
            continue
        
        indices = df_dual.loc[mask, "row_idx"].values
        color = FIXED_LABEL_COLORS.get(rhythm_label, "gray")
        
        ax.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            c=[color],
            marker="^",
            s=50,
            alpha=0.7,
            label=f"Dual: {rhythm_label}+"
        )
    
    # Plot records with no label of interest
    if not df_no_interest.empty:
        indices = df_no_interest["row_idx"].values
        ax.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            c="lightgray",
            marker=".",
            s=10,
            alpha=0.3,
            label="No label of interest"
        )
    
    # Add legend with concise labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", title="Labels")
    
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

# Plot for baseline and fine-tuned embeddings
plot_embedding(axes[0], baseline_umap, df_single_rhythm, df_dual_rhythm, "Baseline (Pre-trained) Model")
plot_embedding(axes[1], finetuned_umap, df_single_rhythm, df_dual_rhythm, "Fine-tuned (Chapman) Model")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("ptbxl_umap_embeddings.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Create Faceted Plots for Each Rhythm Label

# %%
def create_faceted_plots(
    df_single,
    df_dual,
    df_no_interest,
    umap_data,
    title="Faceted Plots by Rhythm Label",
    figsize=(15, 12)
):
    """
    Create faceted plots for each rhythm label, with separate markers for single and dual labeled records.
    
    Parameters
    ----------
    df_single : pd.DataFrame
        DataFrame with single-labeled records
    df_dual : pd.DataFrame
        DataFrame with dual-labeled records
    df_no_interest : pd.DataFrame
        DataFrame with records not of interest
    umap_data : np.ndarray
        2D UMAP embeddings
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with faceted plots
    """
    # Get unique rhythm labels
    rhythm_labels = sorted(LABELS_OF_INTEREST)
    
    # Set up figure and axes
    n_labels = len(rhythm_labels)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each rhythm label in its own facet
    for i, lbl in enumerate(rhythm_labels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot all points in light gray as background
        ax.scatter(
            umap_data[:, 0],
            umap_data[:, 1],
            color="lightgray",
            edgecolor="none",
            s=20,
            alpha=0.2
        )
        
        # Get single-labeled records for this rhythm
        single_lbl = df_single[df_single["rhythm_label"] == lbl]
        
        if len(single_lbl) > 0:
            # Get indices and points for single-labeled records
            single_indices = single_lbl["row_idx"].values
            single_points = np.array([umap_data[idx] for idx in single_indices])
            
            # Plot single-labeled records as circles
            ax.scatter(
                single_points[:, 0],
                single_points[:, 1],
                color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                marker="o",
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                label=f"Single: {lbl}"
            )
        
        # Get dual-labeled records for this rhythm
        dual_lbl = df_dual[df_dual["rhythm_label"] == lbl]
        
        if len(dual_lbl) > 0:
            # Get indices and points for dual-labeled records
            dual_indices = dual_lbl["row_idx"].values
            dual_points = np.array([umap_data[idx] for idx in dual_indices])
            
            # Plot dual-labeled records as triangles
            ax.scatter(
                dual_points[:, 0],
                dual_points[:, 1],
                color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                marker="^",
                s=100,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                label=f"Dual: {lbl}+"
            )
        
        # Set title and labels
        ax.set_title(f"{lbl} Records", fontsize=14)
        ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
        ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
        
        # Create divider for marginal plots
        divider = make_axes_locatable(ax)
        
        # Add top marginal plot (x-dimension)
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Add right marginal plot (y-dimension)
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Add KDE to top marginal plot
        if len(single_points) >= 5:
            try:
                sns.kdeplot(
                    x=single_points[:, 0], 
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                    ax=ax_top
                )
            except Exception:
                pass
                    
        if len(dual_points) >= 2:
            try:
                sns.kdeplot(
                    x=dual_points[:, 0], 
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                    linestyle="--",
                    ax=ax_top
                )
            except Exception:
                pass
                
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        
        # Add KDE to right marginal plot
        if len(single_points) >= 5:
            try:
                sns.kdeplot(
                    y=single_points[:, 1], 
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"), 
                    ax=ax_right
                )
            except Exception:
                pass
                    
        if len(dual_points) >= 2:
            try:
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
            
        # Add individual legend to each facet with concise labels
        if len(single_lbl) > 0 or len(dual_lbl) > 0:
            ax.legend(loc="upper left", fontsize=8)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(title, fontsize=16, y=0.98)
    
    return fig

# Create faceted plots for baseline and fine-tuned embeddings
baseline_faceted = create_faceted_plots(
    df_single_rhythm, 
    df_dual_rhythm,
    df_no_interest,
    baseline_umap,
    "Baseline (Pre-trained) Model - Faceted by Rhythm Label",
    figsize=(15, 12)
)
if baseline_faceted:
    baseline_faceted.savefig("ptbxl_baseline_faceted.png", dpi=300, bbox_inches="tight")

finetuned_faceted = create_faceted_plots(
    df_single_rhythm, 
    df_dual_rhythm,
    df_no_interest,
    finetuned_umap,
    "Fine-tuned (Chapman) Model - Faceted by Rhythm Label",
    figsize=(15, 12)
)
if finetuned_faceted:
    finetuned_faceted.savefig("ptbxl_finetuned_faceted.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## Create Metadata Faceted Plots

# %%
def create_metadata_faceted_plots(
    df_with_metadata,
    umap_data,
    metadata_col,
    title="Faceted by Rhythm Label, Colored by Metadata",
    continuous=False,
    figsize=(15, 12),
    cmap_name="viridis"
):
    """
    Create faceted plots for each rhythm label, with points colored by metadata.
    
    Parameters
    ----------
    df_with_metadata : pd.DataFrame
        DataFrame with metadata and rhythm_label columns
    umap_data : np.ndarray
        2D UMAP embeddings
    metadata_col : str
        Column name for metadata to color by
    title : str
        Plot title
    continuous : bool
        Whether the metadata is continuous or categorical
    figsize : tuple
        Figure size
    cmap_name : str
        Colormap name for continuous data
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with faceted plots
    """
    if "rhythm_label" not in df_with_metadata.columns:
        print(f"No rhythm_label column in records for {metadata_col} plot")
        return None
    
    # Get unique rhythm labels
    rhythm_labels = df_with_metadata["rhythm_label"].dropna().unique()
    rhythm_labels = sorted([lbl for lbl in rhythm_labels if lbl != "Other"])
    
    if len(rhythm_labels) == 0:
        print(f"No rhythm labels found for {metadata_col} plot")
        return None
    
    # Set up figure and axes
    n_labels = len(rhythm_labels)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Set up colormap for continuous data
    if continuous:
        valid_meta = df_with_metadata[metadata_col].dropna()
        
        # Cap age at 90 for visualization purposes
        if metadata_col.lower() == 'age':
            valid_meta = valid_meta.clip(upper=90)
            
        if len(valid_meta) > 0:
            vmin = valid_meta.min()
            vmax = valid_meta.max()
        else:
            vmin, vmax = 0, 1
            
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
    else:
        # For categorical data, get unique values and assign colors
        unique_values = df_with_metadata[metadata_col].dropna().unique()
        unique_values = sorted(unique_values)
        
        # Create a color palette with enough colors - use the same palette as in phase2.py
        palette = sns.color_palette("Set1", n_colors=len(unique_values))
        value_to_color = dict(zip(unique_values, palette))
    
    # Create faceted plots
    for i, lbl in enumerate(rhythm_labels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get records for this rhythm label
        df_rhythm = df_with_metadata[df_with_metadata["rhythm_label"] == lbl]
        
        if len(df_rhythm) == 0:
            ax.text(0.5, 0.5, f"No {lbl} records with {metadata_col} data",
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Plot all points in light gray as background
        ax.scatter(
            umap_data[:, 0],
            umap_data[:, 1],
            color="lightgray",
            edgecolor="none",
            s=20,
            alpha=0.2
        )
        
        # Plot points colored by metadata
        indices = df_rhythm["row_idx"].values
        points = np.array([umap_data[idx] for idx in indices])
        
        if continuous:
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=df_rhythm[metadata_col],
                cmap=cmap,
                marker="o",
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5
            )
        else:
            # For categorical data, plot each category separately
            for value in unique_values:
                value_mask = df_rhythm[metadata_col] == value
                if value_mask.sum() == 0:
                    continue
                
                value_indices = df_rhythm.loc[value_mask, "row_idx"].values
                value_points = np.array([umap_data[idx] for idx in value_indices])
                color = value_to_color[value]
                
                ax.scatter(
                    value_points[:, 0],
                    value_points[:, 1],
                    c=[color],
                    marker="o",
                    s=80,
                    alpha=0.7,
                    edgecolors="white",
                    linewidth=0.5,
                    label=str(value)
                )
            
            # Add legend for categorical data - individual legend for each facet
            ax.legend(title=metadata_col, loc="best", fontsize=8)
        
        # Set title and labels
        ax.set_title(f"{lbl} Records", fontsize=14)
        
        # Only add x-label for the bottom plots
        ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
        ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")
        
        # Create divider for marginal plots
        divider = make_axes_locatable(ax)
        
        # Add top marginal plot (x-dimension)
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Add right marginal plot (y-dimension)
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # For continuous data, we can add KDE plots for the distribution
        if len(points) >= 5:
            try:
                sns.kdeplot(
                    x=points[:, 0], 
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                    ax=ax_top
                )
                sns.kdeplot(
                    y=points[:, 1], 
                    color=FIXED_LABEL_COLORS.get(lbl, "gray"),
                    ax=ax_right
                )
            except Exception:
                pass
                
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        ax_right.set_xlabel("")
        ax_right.set_xticks([])
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add a colorbar for continuous data
    if continuous:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(metadata_col)
    
    # Add overall title
    fig.suptitle(f"{title}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9 if continuous else 1, 0.96])
    
    return fig

# Create faceted plots for age metadata
print("Creating faceted plots for age metadata...")

# Cap age at 90 for visualization
df_records_age = df_records.copy()
df_records_age["age"] = df_records_age["age"].clip(upper=90)

baseline_age_faceted = create_metadata_faceted_plots(
    df_records_age,
    baseline_umap,
    metadata_col="age",
    title="Baseline Model",
    continuous=True,
    cmap_name="viridis",
)
if baseline_age_faceted:
    baseline_age_faceted.savefig("ptbxl_baseline_age_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

finetuned_age_faceted = create_metadata_faceted_plots(
    df_records_age,
    finetuned_umap,
    metadata_col="age",
    title="Fine-tuned Model",
    continuous=True,
    cmap_name="viridis",
)
if finetuned_age_faceted:
    finetuned_age_faceted.savefig("ptbxl_finetuned_age_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

# Create faceted plots for sex metadata
print("Creating faceted plots for sex metadata...")
baseline_sex_faceted = create_metadata_faceted_plots(
    df_records,
    baseline_umap,
    metadata_col="sex",
    title="Baseline Model",
    continuous=False,
)
if baseline_sex_faceted:
    baseline_sex_faceted.savefig("ptbxl_baseline_sex_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

finetuned_sex_faceted = create_metadata_faceted_plots(
    df_records,
    finetuned_umap,
    metadata_col="sex",
    title="Fine-tuned Model",
    continuous=False,
)
if finetuned_sex_faceted:
    finetuned_sex_faceted.savefig("ptbxl_finetuned_sex_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

# Create faceted plots for site metadata
print("Creating faceted plots for site metadata...")
baseline_site_faceted = create_metadata_faceted_plots(
    df_records,
    baseline_umap,
    metadata_col="site",
    title="Baseline Model",
    continuous=False,
)
if baseline_site_faceted:
    baseline_site_faceted.savefig("ptbxl_baseline_site_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

finetuned_site_faceted = create_metadata_faceted_plots(
    df_records,
    finetuned_umap,
    metadata_col="site",
    title="Fine-tuned Model",
    continuous=False,
)
if finetuned_site_faceted:
    finetuned_site_faceted.savefig("ptbxl_finetuned_site_faceted.png", dpi=300, bbox_inches="tight")
    plt.show()

print("Analysis complete. All plots have been saved.")