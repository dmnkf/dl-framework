#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 (CMR - ACDC): Embedding Space Analysis (Pre-trained vs Fine-tuned)
--------------------------------------------------------------------------
We load the ACDC dataset (single-label, multi-class), run UMAP on Baseline
vs. Fine-tuned embeddings, and visualize the distribution of classes.
Now using k-fold validation splits and including BMI analysis.
"""

# %% [markdown]
# # ACDC CMR: Phase 1 Analysis
#
# This notebook-style script replicates the logic of the ECG Phase 1
# but for the simpler ACDC CMR dataset, which is single-label multi-class.
# Now using k-fold validation splits and including BMI analysis.

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Adjust paths as necessary for your project
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
# We define fixed colors for specific labels.

# %%
# Define a color palette for our classes
SINGLE_COLOR_PALETTE = sns.color_palette("colorblind", 5)

# Fixed colors for specific CMR classes if needed
# not used anymore
FIXED_LABEL_COLORS = {
}

# %% [markdown]
# ## Data Loading
# Here, we load the **ACDC CMR dataset** from our `UnifiedDataset`.
# We then retrieve embeddings for **Baseline** and **Fine-Tuned** models.

# %%
print("Phase 1 (CMR): UMAP on ACDC data, comparing baseline vs fine-tuned embeddings.")
print("=" * 70)

data_root = project_root / "data"  # Adjust if needed
acdc_data = UnifiedDataset(data_root, modality=DatasetModality.CMR, dataset_key="acdc")
acdc_md_store = acdc_data.metadata_store

acdc_splits = acdc_data.get_splits()

# %% [markdown]
# ## Helper Functions for BMI Analysis

# %%
def calculate_bmi(height_cm, weight_kg):
    """
    Calculate BMI from height (in cm) and weight (in kg).
    BMI = weight (kg) / (height (m))^2
    """
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    
    # Convert height from cm to m
    height_m = height_cm / 100.0
    
    # Calculate BMI
    bmi = weight_kg / (height_m * height_m)
    
    # Return None for unrealistic BMI values
    if bmi < 10 or bmi > 60:
        return None
        
    return bmi

def get_bmi_category(bmi):
    """
    Get BMI category based on BMI value.
    """
    if bmi is None:
        return None
    
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def plot_embedding_with_bmi(ax, emb_2d, df_records, title, continuous=True):
    """
    Plot UMAP embedding colored by BMI.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    emb_2d : ndarray
        2D UMAP embedding
    df_records : DataFrame
        DataFrame containing record information with BMI data
    title : str
        Title for the plot
    continuous : bool
        If True, use continuous color scale for BMI
        If False, use categorical colors for BMI categories
    """
    ax.set_title(title, fontsize=16)
    
    # Filter records with valid BMI
    df_valid_bmi = df_records[df_records["bmi"].notna()].copy()
    
    if continuous:
        # For continuous BMI values
        norm = Normalize(vmin=15, vmax=40)  # Reasonable BMI range
        cmap = plt.get_cmap("viridis")
        
        for row in df_valid_bmi.itertuples():
            row_idx = getattr(row, "row_idx")
            bmi = getattr(row, "bmi")
            
            ax.scatter(
                emb_2d[row_idx, 0],
                emb_2d[row_idx, 1],
                c=[cmap(norm(bmi))],
                s=80,
                alpha=0.9,
                edgecolors="white",
                linewidth=0.5,
            )
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("BMI", fontsize=12)
    else:
        # For categorical BMI values
        df_valid_bmi["bmi_category"] = df_valid_bmi["bmi"].apply(get_bmi_category)
        
        # Define colors for BMI categories using a colorblind-friendly palette (viridis-based)
        bmi_category_colors = {
            "Underweight": "#440154",  # Dark purple (viridis start)
            "Normal": "#21918c",       # Teal (viridis middle)
            "Overweight": "#90d743",   # Light green (viridis towards end)
            "Obese": "#fde725"         # Yellow (viridis end)
        }
        
        # Define markers for BMI categories
        bmi_category_markers = {
            "Underweight": "o",  # Circle
            "Normal": "s",       # Square
            "Overweight": "^",   # Triangle up
            "Obese": "D"         # Diamond
        }
        
        # Define BMI thresholds for legend
        bmi_thresholds = {
            "Underweight": "<18.5",
            "Normal": "18.5-24.9",
            "Overweight": "25-29.9",
            "Obese": "≥30"
        }
        
        # Plot each BMI category
        for category, color in bmi_category_colors.items():
            category_rows = df_valid_bmi[df_valid_bmi["bmi_category"] == category]
            
            if len(category_rows) > 0:
                category_indices = category_rows["row_idx"].values
                
                ax.scatter(
                    emb_2d[category_indices, 0],
                    emb_2d[category_indices, 1],
                    c=color,
                    s=80,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                    marker=bmi_category_markers[category],
                    label=f"{category} (BMI {bmi_thresholds[category]}, n={len(category_rows)})"
                )
        
        # Add legend with dynamic position
        # Try different positions and use the one with least overlap
        ax.legend(
            fontsize=10,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
        )
    
    ax.set_xlabel("UMAP Dim 1", fontsize=14)
    ax.set_ylabel("UMAP Dim 2", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)

def create_bmi_faceted_plots(df_records, umap_data, title="BMI Analysis", figsize=(15, 12)):
    """
    Create faceted plots for each class label, colored by BMI.
    
    Parameters:
    -----------
    df_records : DataFrame
        DataFrame containing record information with BMI data
    umap_data : ndarray
        UMAP embeddings for all records
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Filter records with valid BMI
    df_valid = df_records[df_records["bmi"].notna()].copy()
    
    if len(df_valid) == 0:
        print(f"No records with valid BMI data")
        return None
    
    # Get unique class labels
    unique_labels = sorted(df_valid["class_label"].unique())
    
    # Generate a color map for each unique class label
    class_to_color = {}
    
    # First, use fixed colors for specific labels if defined
    for lbl in unique_labels:
        if lbl in FIXED_LABEL_COLORS:
            class_to_color[lbl] = FIXED_LABEL_COLORS[lbl]
    
    # Then assign colors to remaining labels
    color_idx = 0
    for lbl in unique_labels:
        if lbl not in class_to_color:
            # Skip any colors used in FIXED_LABEL_COLORS
            while color_idx < len(SINGLE_COLOR_PALETTE) and any(
                SINGLE_COLOR_PALETTE[color_idx] == c for c in FIXED_LABEL_COLORS.values()
            ):
                color_idx += 1
    
            if color_idx < len(SINGLE_COLOR_PALETTE):
                class_to_color[lbl] = SINGLE_COLOR_PALETTE[color_idx]
                color_idx += 1
            else:
                # fallback color
                class_to_color[lbl] = (0.5, 0.5, 0.5)
    
    n_labels = len(unique_labels)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Set up color normalization for BMI
    norm = Normalize(vmin=15, vmax=40)  # Reasonable BMI range
    cmap = plt.get_cmap("viridis")
    
    for i, lbl in enumerate(unique_labels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter records for this class label
        df_class = df_valid[df_valid["class_label"] == lbl].copy()
        
        if len(df_class) < 3:
            ax.text(
                0.5,
                0.5,
                f"Not enough data for {lbl}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Class: {lbl}", fontsize=14)
            ax.axis("off")
            continue
        
        # Get UMAP coordinates for this class
        class_points = np.array([umap_data[idx] for idx in df_class["row_idx"]])
        
        # Create a DataFrame for the points for easier KDE plotting
        df_class_points = pd.DataFrame({
            'x': class_points[:, 0],
            'y': class_points[:, 1],
            'bmi': df_class["bmi"].values
        })

        # Plot points colored by BMI
        scatter = ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            c=df_class["bmi"].values,
            cmap=cmap,
            norm=norm,
            s=80,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        
        # Create divider for marginal plots
        divider = make_axes_locatable(ax)
        
        # Add top marginal plot (x-dimension)
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        
        # Set the x-limits to match the main plot
        ax_top.set_xlim(ax.get_xlim())
        
        # Plot KDE on the marginal axes with class-specific color
        if len(df_class_points) >= 3:
            sns.kdeplot(
                data=df_class_points, x="x", ax=ax_top, 
                color=class_to_color.get(lbl, (0.5, 0.5, 0.5)), fill=True, alpha=0.3, linewidth=1.5
            )
        
        # Add right marginal plot (y-dimension)
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        
        # Set the y-limits to match the main plot
        ax_right.set_ylim(ax.get_ylim())
        
        # Plot KDE on the marginal axes with class-specific color
        if len(df_class_points) >= 3:
            sns.kdeplot(
                data=df_class_points, y="y", ax=ax_right, 
                color=class_to_color.get(lbl, (0.5, 0.5, 0.5)), fill=True, alpha=0.3, linewidth=1.5
            )
        
        # Set the facet title on the top axis instead of the main axis
        ax_top.set_title(f"Class: {lbl} (n={len(df_class)})", fontsize=14, pad=15)
        
        # No title on the main axis
        ax.set_title("")
        
        ax.set_xlabel("UMAP Dim 1", fontsize=12)
        ax.set_ylabel("UMAP Dim 2", fontsize=12)
    
    # Hide any unused subplots
    for i in range(len(unique_labels), len(axes)):
        axes[i].axis('off')
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("BMI", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig

def create_bmi_category_faceted_plots(df_records, umap_data, title="BMI Category Analysis", figsize=(15, 12)):
    """
    Create faceted plots for each class label, with points colored by BMI category.
    
    Parameters:
    -----------
    df_records : DataFrame
        DataFrame containing record information with BMI data
    umap_data : ndarray
        UMAP embeddings for all records
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Filter records with valid BMI
    df_valid = df_records[df_records["bmi"].notna()].copy()
    
    if len(df_valid) == 0:
        print(f"No records with valid BMI data")
        return None
    
    # Add BMI category to the DataFrame
    df_valid["bmi_category"] = df_valid["bmi"].apply(get_bmi_category)
    
    # Define colors for BMI categories using a colorblind-friendly palette (viridis-based)
    bmi_category_colors = {
        "Underweight": "#440154",  # Dark purple (viridis start)
        "Normal": "#21918c",       # Teal (viridis middle)
        "Overweight": "#90d743",   # Light green (viridis towards end)
        "Obese": "#fde725"         # Yellow (viridis end)
    }
    
    # Define markers for BMI categories
    bmi_category_markers = {
        "Underweight": "o",  # Circle
        "Normal": "s",       # Square
        "Overweight": "^",   # Triangle up
        "Obese": "D"         # Diamond
    }
    
    # Define BMI thresholds for legend
    bmi_thresholds = {
        "Underweight": "<18.5",
        "Normal": "18.5-24.9",
        "Overweight": "25-29.9",
        "Obese": "≥30"
    }
    
    # Get unique class labels
    unique_labels = sorted(df_valid["class_label"].unique())
    
    # Generate class colors for KDE plots
    class_to_color = {}
    
    # First, use fixed colors for specific labels if defined
    for lbl in unique_labels:
        if lbl in FIXED_LABEL_COLORS:
            class_to_color[lbl] = FIXED_LABEL_COLORS[lbl]
    
    # Then assign colors to remaining labels
    color_idx = 0
    for lbl in unique_labels:
        if lbl not in class_to_color:
            # Skip any colors used in FIXED_LABEL_COLORS
            while color_idx < len(SINGLE_COLOR_PALETTE) and any(
                SINGLE_COLOR_PALETTE[color_idx] == c for c in FIXED_LABEL_COLORS.values()
            ):
                color_idx += 1
    
            if color_idx < len(SINGLE_COLOR_PALETTE):
                class_to_color[lbl] = SINGLE_COLOR_PALETTE[color_idx]
                color_idx += 1
            else:
                # fallback color
                class_to_color[lbl] = (0.5, 0.5, 0.5)
    
    n_labels = len(unique_labels)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, lbl in enumerate(unique_labels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get class color for KDE plots
        class_color = class_to_color.get(lbl, (0.5, 0.5, 0.5))
        
        # Filter records for this class label
        df_class = df_valid[df_valid["class_label"] == lbl].copy()
        
        if len(df_class) < 3:
            ax.text(
                0.5,
                0.5,
                f"Not enough data for {lbl}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Class: {lbl}", fontsize=14)
            ax.axis("off")
            continue
        
        # Get UMAP coordinates for this class
        class_points = np.array([umap_data[idx] for idx in df_class["row_idx"]])
        
        # Create a DataFrame for the points for easier KDE plotting
        df_class_points = pd.DataFrame({
            'x': class_points[:, 0],
            'y': class_points[:, 1],
            'bmi': df_class["bmi"].values,
            'bmi_category': df_class["bmi_category"].values
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

        # Plot points by BMI category
        for category, color in bmi_category_colors.items():
            category_points = df_class_points[df_class_points["bmi_category"] == category]
            
            if len(category_points) > 0:
                category_indices = [df_class["row_idx"].iloc[i] for i in category_points.index]
                
                ax.scatter(
                    category_points['x'],
                    category_points['y'],
                    c=color,
                    s=80,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                    marker=bmi_category_markers[category],
                    label=f"{category} (BMI {bmi_thresholds[category]}, n={len(category_points)})"
                )
        
        # Create divider for marginal plots
        divider = make_axes_locatable(ax)
        
        # Add top marginal plot (x-dimension)
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        
        # Set the x-limits to match the main plot
        ax_top.set_xlim(ax.get_xlim())
        
        # Plot KDE on the marginal axes with class-specific color
        if len(df_class_points) >= 3:
            sns.kdeplot(
                data=df_class_points, x="x", ax=ax_top, 
                color=class_color, fill=True, alpha=0.3, linewidth=1.5
            )
        
        # Add right marginal plot (y-dimension)
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        
        # Set the y-limits to match the main plot
        ax_right.set_ylim(ax.get_ylim())
        
        # Plot KDE on the marginal axes with class-specific color
        if len(df_class_points) >= 3:
            sns.kdeplot(
                data=df_class_points, y="y", ax=ax_right, 
                color=class_color, fill=True, alpha=0.3, linewidth=1.5
            )
        
        # Set the facet title on the top axis instead of the main axis
        ax_top.set_title(f"Class: {lbl} (n={len(df_class)})", fontsize=14, pad=15)
        
        # No title on the main axis
        ax.set_title("")
        
        ax.set_xlabel("UMAP Dim 1", fontsize=12)
        ax.set_ylabel("UMAP Dim 2", fontsize=12)
        
        # Add legend for BMI categories
        handles, labels = ax.get_legend_handles_labels()
        # Filter out "All Records" from the legend
        filtered_handles = [h for h, l in zip(handles, labels) if l != "All Records"]
        filtered_labels = [l for l in labels if l != "All Records"]
        
        ax.legend(
            handles=filtered_handles,
            labels=filtered_labels,
            fontsize=8,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
        )
    
    # Hide any unused subplots
    for i in range(len(unique_labels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# %% [markdown]
# ## Process Each Fold
# We'll now create a separate UMAP visualization for each fold.

# %%
# Define the number of folds
num_folds = 5

# Define the embedding types for each fold
fold_embedding_types = {
    0: {"baseline": "baseline_0", "fine_tuned": "fine_tuned_0_epoch_079"},
    1: {"baseline": "baseline_1", "fine_tuned": "fine_tuned_1_epoch_079"},
    2: {"baseline": "baseline_2", "fine_tuned": "fine_tuned_2_epoch_079"},
    3: {"baseline": "baseline_3", "fine_tuned": "fine_tuned_3_epoch_079"},
    4: {"baseline": "baseline_4", "fine_tuned": "fine_tuned_4_epoch_079"},
}

# UMAP parameters (consistent across all folds)
umap_params = dict(n_neighbors=15, n_components=2, metric="euclidean", random_state=42)

# Process each fold
for fold_idx in range(num_folds):
    fold_num = fold_idx + 1  # Fold numbers are 1-indexed
    print(f"\n\n{'='*30} Processing Fold {fold_num} {'='*30}")
    
    # Get validation IDs for this fold
    val_ids = acdc_splits.get(f"fold_{fold_num}_val", [])
    print(f"Number of validation records for fold {fold_num}: {len(val_ids)}")
    print(f"First 5 validation IDs: {val_ids[:5]}")
    
    # Define embedding types for this fold
    embedding_types = {
        fold_embedding_types[fold_idx]["baseline"]: f"Baseline (Fold {fold_num})",
        fold_embedding_types[fold_idx]["fine_tuned"]: f"Fine-tuned (Fold {fold_num})",
    }
    
    # Collect records and embeddings for this fold
    records_info = []
    embeddings_by_type = {emb_type: [] for emb_type in embedding_types.keys()}
    
    # `raw_dataset` grants direct access to the underlying CMR records (with .load_record)
    raw_acdc = acdc_data.raw_dataset
    
    for rid in val_ids:
        try:
            # Load the record from the raw dataset
            record = raw_acdc.load_record(rid)
    
            # Single-label class name; e.g., "MINF", "DCM", etc.
            # target_labels is typically a list but contains only one element
            if isinstance(record.target_labels, list) and len(record.target_labels) == 1:
                class_label = record.target_labels[0]
            else:
                # Fallback if it's already a string or something different
                class_label = record.target_labels
    
            # Get metadata for BMI calculation
            meta = acdc_md_store.get(rid, {})
            height = meta.get("height", None)
            weight = meta.get("weight", None)
            bmi = calculate_bmi(height, weight) if height and weight else None
    
            # Check that all embedding types are available
            all_embeds_available = True
            for emb_type in embedding_types.keys():
                try:
                    emb = acdc_data.get_embeddings(rid, embeddings_type=emb_type)
                    embeddings_by_type[emb_type].append(emb)
                except Exception as e:
                    print(f"Skipping {rid} - missing {emb_type} embedding: {e}")
                    all_embeds_available = False
                    break
            
            if all_embeds_available:
                records_info.append({
                    "record_id": rid, 
                    "class_label": class_label,
                    "height": height,
                    "weight": weight,
                    "bmi": bmi
                })
    
        except Exception as e:
            print(f"Skipping {rid}. Error loading record: {e}")
    
    df_records = pd.DataFrame(records_info)
    if len(df_records) == 0:
        print(f"No records with all embedding types found for fold {fold_num}. Skipping.")
        continue
        
    df_records["row_idx"] = df_records.index
    print(f"Collected {len(df_records)} records with all embedding types for fold {fold_num}.")
    print(f"Records with valid BMI: {df_records['bmi'].notna().sum()} ({df_records['bmi'].notna().sum() / len(df_records) * 100:.1f}%)")
    
    # Stack embeddings
    stacked_embeddings = {}
    for emb_type in embedding_types.keys():
        if embeddings_by_type[emb_type]:
            stacked_embeddings[emb_type] = np.vstack(embeddings_by_type[emb_type])
            print(f"{emb_type} shape:", stacked_embeddings[emb_type].shape)
        else:
            print(f"WARNING: No embeddings collected for {emb_type}")
    
    # Apply UMAP to each embedding type
    umap_projections = {}
    for emb_type, embeddings in stacked_embeddings.items():
        print(f"Running UMAP on {emb_type}...")
        umap_projections[emb_type] = run_umap(embeddings, **umap_params)
    
    # Generate a color map for each unique class label
    unique_labels = sorted(df_records["class_label"].unique())
    class_to_color = {}
    
    # First, use fixed colors for specific labels if defined
    for lbl in unique_labels:
        if lbl in FIXED_LABEL_COLORS:
            class_to_color[lbl] = FIXED_LABEL_COLORS[lbl]
    
    # Then assign colors to remaining labels
    color_idx = 0
    for lbl in unique_labels:
        if lbl not in class_to_color:
            # Skip any colors used in FIXED_LABEL_COLORS
            while color_idx < len(SINGLE_COLOR_PALETTE) and any(
                SINGLE_COLOR_PALETTE[color_idx] == c for c in FIXED_LABEL_COLORS.values()
            ):
                color_idx += 1
    
            if color_idx < len(SINGLE_COLOR_PALETTE):
                class_to_color[lbl] = SINGLE_COLOR_PALETTE[color_idx]
                color_idx += 1
            else:
                # fallback color
                class_to_color[lbl] = (0.5, 0.5, 0.5)
    
    # Determine grid size based on number of embedding types
    num_embeddings = len(embedding_types)
    if num_embeddings <= 3:
        # 1 row, up to 3 columns
        nrows, ncols = 1, num_embeddings
    elif num_embeddings <= 6:
        # 2 rows, up to 3 columns
        nrows, ncols = 2, (num_embeddings + 1) // 2
    else:
        # 3 rows, variable columns
        nrows = 3
        ncols = (num_embeddings + 2) // 3
    
    # Create figure with a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8 + 1, nrows * 6))
    # fig.suptitle(
    #     f"ACDC CMR: Fold {fold_num} - Embedding Comparison", fontsize=20, fontweight="bold", y=0.98
    # )
    
    def plot_embedding(ax, emb_2d, title):
        ax.set_title(title, fontsize=16)
    
        # Define markers for different classes to improve colorblind friendliness
        class_markers = {
            "DCM": "o",      # Circle
            "HCM": "s",      # Square
            "MINF": "^",     # Triangle up
            "NOR": "D",      # Diamond
            "RV": "P",       # Plus (filled)
        }
        
        # Default marker if class not in the dictionary
        default_marker = "o"
    
        # (1) All points in light gray
        ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            color="lightgray",
            edgecolor="none",
            s=40,
            alpha=0.6,
            label="All Records",
            marker="o",
        )
    
        # (2) Overlay points with class colors and specific markers
        for row in df_records.itertuples():
            row_idx = getattr(row, "row_idx")
            class_label = getattr(row, "class_label")
            color = class_to_color.get(class_label, (0.5, 0.5, 0.5))
            marker = class_markers.get(class_label, default_marker)
    
            ax.scatter(
                emb_2d[row_idx, 0],
                emb_2d[row_idx, 1],
                c=[color],
                s=80,
                alpha=0.9,
                edgecolors="white",
                linewidth=0.5,
                marker=marker,
            )
    
        ax.set_xlabel("UMAP Dim 1", fontsize=14)
        ax.set_ylabel("UMAP Dim 2", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add legend for class colors and markers
        handles = []
        for class_label in sorted(class_to_color.keys()):
            handles.append(
                Line2D(
                    [0], [0],
                    marker=class_markers.get(class_label, default_marker),
                    color="w",
                    markerfacecolor=class_to_color.get(class_label, (0.5, 0.5, 0.5)),
                    markersize=10,
                    label=class_label,
                    linewidth=0,
                )
            )
        
        # Place legend outside the plot
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot embeddings
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each embedding type in its own subplot
    for i, (emb_type, title) in enumerate(embedding_types.items()):
        if i < len(axes_flat) and emb_type in umap_projections:
            plot_embedding(axes_flat[i], umap_projections[emb_type], f"{title}")
            
    # Hide any unused subplots
    for i in range(len(embedding_types), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(
        f"results/cmr_embedding_fold_{fold_num}.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    
    # Now create BMI-colored plots for each embedding type
    for emb_type, title in embedding_types.items():
        if emb_type in umap_projections:
            # Create faceted BMI plots
            fig = create_bmi_faceted_plots(
                df_records,
                umap_projections[emb_type],
                f"{title} - Faceted by Class, Colored by BMI"
            )
            if fig:
                plt.savefig(
                    f"cmr_embedding_fold_{fold_num}_{emb_type}_bmi_faceted.png", 
                    dpi=150, 
                    bbox_inches="tight"
                )
                plt.show()
            
            # Create faceted BMI category plots
            fig = create_bmi_category_faceted_plots(
                df_records,
                umap_projections[emb_type],
                f"{title} - Faceted by Class, Colored by BMI Category"
            )
            if fig:
                plt.savefig(
                    f"results/cmr_embedding_fold_{fold_num}_{emb_type}_bmi_category_faceted.png", 
                    dpi=150, 
                    bbox_inches="tight"
                )
                plt.show()

# %% [markdown]
# ## Evolution of Embedding Spaces
# Now we'll create evolution plots showing how the embedding spaces change with more training.
# For each fold, we'll compare baseline, fine_tuned_039, fine_tuned_079, and fine_tuned_149.

# %%
print("\n\n" + "="*50)
print("Creating embedding space evolution plots for each fold")
print("="*50)

# Evolution embedding types for each fold
evolution_embedding_types = {
    0: {
        "baseline": "baseline_0", 
        "fine_tuned_039": "fine_tuned_0_epoch_039",
        "fine_tuned_079": "fine_tuned_0_epoch_079",
        "fine_tuned_149": "fine_tuned_0_epoch_149"
    },
    1: {
        "baseline": "baseline_1", 
        "fine_tuned_039": "fine_tuned_1_epoch_039",
        "fine_tuned_079": "fine_tuned_1_epoch_079",
        "fine_tuned_149": "fine_tuned_1_epoch_149"
    },
    2: {
        "baseline": "baseline_2", 
        "fine_tuned_039": "fine_tuned_2_epoch_039",
        "fine_tuned_079": "fine_tuned_2_epoch_079",
        "fine_tuned_149": "fine_tuned_2_epoch_149"
    },
    3: {
        "baseline": "baseline_3", 
        "fine_tuned_039": "fine_tuned_3_epoch_039",
        "fine_tuned_079": "fine_tuned_3_epoch_079",
        "fine_tuned_149": "fine_tuned_3_epoch_149"
    },
    4: {
        "baseline": "baseline_4", 
        "fine_tuned_039": "fine_tuned_4_epoch_039",
        "fine_tuned_079": "fine_tuned_4_epoch_079",
        "fine_tuned_149": "fine_tuned_4_epoch_149"
    },
}

# Process each fold
for fold_idx in range(num_folds):
    fold_num = fold_idx + 1  # Fold numbers are 1-indexed
    print(f"\n\n{'='*30} Processing Evolution for Fold {fold_num} {'='*30}")
    
    # Get validation IDs for this fold
    val_ids = acdc_splits[f"fold_{fold_num}_val"]
    print(f"Validation set size for fold {fold_num}: {len(val_ids)}")
    
    # Collect record info and embeddings for each embedding type
    records_info = []
    embeddings_by_type = {
        "baseline": [],
        "fine_tuned_039": [],
        "fine_tuned_079": [],
        "fine_tuned_149": []
    }
    
    # `raw_dataset` grants direct access to the underlying CMR records (with .load_record)
    raw_acdc = acdc_data.raw_dataset
    
    for rid in val_ids:
        try:
            # Load the record from the raw dataset
            record = raw_acdc.load_record(rid)
    
            # Single-label class name; e.g., "MINF", "DCM", etc.
            # target_labels is typically a list but contains only one element
            if isinstance(record.target_labels, list) and len(record.target_labels) == 1:
                class_label = record.target_labels[0]
            else:
                # Fallback if it's already a string or something different
                class_label = record.target_labels
    
            # Get metadata for BMI calculation
            meta = acdc_md_store.get(rid, {})
            height = meta.get("height", None)
            weight = meta.get("weight", None)
            
            # Calculate BMI if height and weight are available
            bmi = None
            if height is not None and weight is not None:
                bmi = calculate_bmi(height, weight)
        
            # Check if all embedding types are available
            all_embeds_available = True
            
            # Get embeddings for each type
            for emb_type, emb_name in evolution_embedding_types[fold_idx].items():
                try:
                    emb = acdc_data.get_embeddings(rid, embeddings_type=emb_name)
                    embeddings_by_type[emb_type].append(emb)
                except Exception as e:
                    print(f"Missing {emb_type} embedding for {rid}: {e}")
                    all_embeds_available = False
                    break
            
            if all_embeds_available:
                records_info.append({
                    "record_id": rid, 
                    "class_label": class_label,
                    "height": height,
                    "weight": weight,
                    "bmi": bmi
                })
        
        except Exception as e:
            print(f"Skipping {rid}. Error loading record: {e}")
    
    df_records = pd.DataFrame(records_info)
    if len(df_records) == 0:
        print(f"No records with all embedding types found for fold {fold_num}. Skipping.")
        continue
        
    df_records["row_idx"] = df_records.index
    print(f"Collected {len(df_records)} records with all embedding types for fold {fold_num}.")
    
    # Stack embeddings
    stacked_embeddings = {}
    for emb_type in embeddings_by_type.keys():
        if embeddings_by_type[emb_type]:
            stacked_embeddings[emb_type] = np.vstack(embeddings_by_type[emb_type])
            print(f"{emb_type} shape:", stacked_embeddings[emb_type].shape)
        else:
            print(f"WARNING: No embeddings collected for {emb_type}")
    
    # Apply UMAP to each embedding type
    umap_projections = {}
    for emb_type, embeddings in stacked_embeddings.items():
        print(f"Running UMAP on {emb_type}...")
        umap_projections[emb_type] = run_umap(embeddings, **umap_params)
    
    # Generate a color map for each unique class label
    unique_labels = sorted(df_records["class_label"].unique())
    class_to_color = {}
    
    # First, use fixed colors for specific labels if defined
    for lbl in unique_labels:
        if lbl in FIXED_LABEL_COLORS:
            class_to_color[lbl] = FIXED_LABEL_COLORS[lbl]
    
    # Then assign colors to remaining labels
    color_idx = 0
    for lbl in unique_labels:
        if lbl not in class_to_color:
            # Skip any colors used in FIXED_LABEL_COLORS
            while color_idx < len(SINGLE_COLOR_PALETTE) and any(
                SINGLE_COLOR_PALETTE[color_idx] == c for c in FIXED_LABEL_COLORS.values()
            ):
                color_idx += 1
            
            # Assign color
            if color_idx < len(SINGLE_COLOR_PALETTE):
                class_to_color[lbl] = SINGLE_COLOR_PALETTE[color_idx]
                color_idx += 1
            else:
                # Fallback to a default color if we run out of colors
                class_to_color[lbl] = (0.5, 0.5, 0.5)  # Gray
    
    # Determine grid layout based on number of embeddings
    num_embeddings = len([et for et in evolution_embedding_types[fold_idx].keys() if et in umap_projections])
    
    if num_embeddings <= 3:
        # 1 row, up to 3 columns
        nrows, ncols = 1, num_embeddings
    elif num_embeddings <= 6:
        # 2 rows, up to 3 columns
        nrows, ncols = 2, min(3, (num_embeddings + 1) // 2)
    else:
        # 3 rows, up to 3 columns
        nrows, ncols = 3, min(3, (num_embeddings + 2) // 3)
    
    # Create figure with a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8 + 1, nrows * 6))
    
    def plot_embedding(ax, emb_2d, title):
        ax.set_title(title, fontsize=16)
        
        # Define markers for different classes to improve colorblind friendliness
        class_markers = {
            "DCM": "o",      # Circle
            "HCM": "s",      # Square
            "MINF": "^",     # Triangle up
            "NOR": "D",      # Diamond
            "RV": "P",       # Plus (filled)
        }
        
        # Default marker if class not in the dictionary
        default_marker = "o"
    
        # Plot each class with its own color and marker
        for lbl in unique_labels:
            mask = df_records["class_label"] == lbl
            if mask.any():
                color = class_to_color[lbl]
                marker = class_markers.get(lbl, default_marker)
                ax.scatter(
                    emb_2d[mask.values, 0],
                    emb_2d[mask.values, 1],
                    c=[color],
                    s=60,
                    alpha=0.7,
                    label=lbl,
                    edgecolors="white",
                    linewidth=0.5,
                    marker=marker,
                )
        
        ax.set_xlabel("UMAP Dim 1", fontsize=14)
        ax.set_ylabel("UMAP Dim 2", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add legend for class colors and markers
        handles = []
        for class_label in sorted(class_to_color.keys()):
            handles.append(
                Line2D(
                    [0], [0],
                    marker=class_markers.get(class_label, default_marker),
                    color="w",
                    markerfacecolor=class_to_color.get(class_label, (0.5, 0.5, 0.5)),
                    markersize=10,
                    label=class_label,
                    linewidth=0,
                )
            )
        
        # Place legend outside the plot
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot embeddings
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each embedding type in its own subplot
    ax_idx = 0
    embedding_titles = {
        "baseline": "Baseline (Pre-trained)",
        "fine_tuned_039": "Fine-tuned (Epoch 40)",
        "fine_tuned_079": "Fine-tuned (Epoch 80)",
        "fine_tuned_149": "Fine-tuned (Epoch 150)"
    }
    
    for emb_type, title in embedding_titles.items():
        if emb_type in umap_projections and ax_idx < len(axes_flat):
            plot_embedding(axes_flat[ax_idx], umap_projections[emb_type], title)
            ax_idx += 1
    
    # Hide any unused subplots
    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(
        f"results/cmr_embedding_evolution_fold_{fold_num}.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

print("\nAll evolution plots created successfully!")

# %%
