#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: PTB-XL Only - Embedding Space Analysis (Pre-trained vs. Fine-tuned)
----------------------------------------------------------------------------
Similar to phase1.py, but applied exclusively to PTB-XL test set.
We show:
  - A 2D UMAP projection of PTB-XL embeddings (baseline and fine-tuned).
  - Color-coded single-rhythm records (exactly 1 scp_code with "rhythm"=1.0).
  - Gray background for all other records.
"""
# %%

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Adjust as needed to match your project structure
project_root = Path().absolute().parent.parent
sys.path.append(str(project_root))

from src.visualization.embedding_viz import run_umap
from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

# --------------------------------------------------------------------------
# Matplotlib / Seaborn style
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 12

# --------------------------------------------------------------------------
# 1) Define color palette, label mappings, and markers
# --------------------------------------------------------------------------

SINGLE_COLOR_PALETTE = sns.color_palette("tab10", 9)

FIXED_LABEL_COLORS = {
    "SR": SINGLE_COLOR_PALETTE[0], 
    "AFIB": SINGLE_COLOR_PALETTE[1],
    "SB": SINGLE_COLOR_PALETTE[3],  
    "GSVT": SINGLE_COLOR_PALETTE[4], 
    "PACE": SINGLE_COLOR_PALETTE[7],  
}


# Mapping PTB-XL scp_codes to arrhythmia-style integration labels
# (Same as in phase3.py, but potentially shortened or adjusted to your needs)
PTBXL_ARR_MAP = {
    "AFIB": "AFIB",
    "AFLT": "AFIB",
    "NORM": "SR",
    "SR": "SR",
    "SARRH": "SR",
    "SBRAD": "SB",
    "STACH": "GSVT",
    "SVTAC": "GSVT",
    "SVARR": "GSVT",
    "PSVT": "GSVT",
    "PACE": "PACE",
}

# You may want to unify the idea of "group" or markers if you prefer
# For simplicity, let's assume all single-labeled rhythms get the same marker.
DEFAULT_MARKER = "o"

# --------------------------------------------------------------------------
# 2) Load PTB-XL data
# --------------------------------------------------------------------------
print("Loading PTB-XL dataset...")

ptbxl_data = UnifiedDataset(
    project_root / "data", modality=DatasetModality.ECG, dataset_key="ptbxl"
)

ptbxl_splits = ptbxl_data.get_splits()
ptbxl_test_ids = ptbxl_splits.get("test", [])
md_store = ptbxl_data.metadata_store

pretrained_embedding = "baseline"  # or whatever key you used
finetuned_embedding = "fine_tuned_50"  # or whatever key for your fine-tuned

records_info = []
emb_list_baseline = []
emb_list_finetuned = []

for rid in ptbxl_test_ids:
    meta = md_store.get(rid, {})
    scp_codes = meta.get("scp_codes", {})
    scp_statements = meta.get("scp_statements", {})

    # Attempt to load both embeddings; skip if missing
    try:
        emb_base = ptbxl_data.get_embeddings(rid, embeddings_type=pretrained_embedding)
        emb_ft = ptbxl_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
    except Exception as e:
        # If embeddings are missing, skip
        continue

    # Identify all scp_codes that are "rhythm=1.0"
    valid_rhythm_codes = []
    for code_key in scp_codes:
        if code_key in scp_statements:
            code_info = scp_statements[code_key]
            if code_info.get("rhythm", 0.0) == 1.0:
                valid_rhythm_codes.append(code_key)

    # We'll store them, then later see if exactly 1 code is present
    # If code in PTBXL_ARR_MAP, we can map it to "SR", "AFIB", etc.
    records_info.append(
        {
            "record_id": rid,
            "emb_base": emb_base,
            "emb_ft": emb_ft,
            "valid_rhythm_codes": valid_rhythm_codes,
        }
    )
    emb_list_baseline.append(emb_base)
    emb_list_finetuned.append(emb_ft)

df_records = pd.DataFrame(records_info)
print(f"Loaded {len(df_records)} PTB-XL records with embeddings.")


# --------------------------------------------------------------------------
# 3) Stack embeddings and run UMAP
# --------------------------------------------------------------------------
# 3a) Convert to arrays for baseline and fine-tuned
baseline_embeddings = np.vstack(df_records["emb_base"].values)
finetuned_embeddings = np.vstack(df_records["emb_ft"].values)

print("Running UMAP on PTB-XL test set (baseline + fine-tuned).")
umap_params = dict(n_neighbors=15, n_components=2, metric="euclidean", random_state=42)

umap_baseline = run_umap(baseline_embeddings, **umap_params)  # shape: (N,2)
umap_finetuned = run_umap(finetuned_embeddings, **umap_params)  # shape: (N,2)

df_records["umap_x_base"] = umap_baseline[:, 0]
df_records["umap_y_base"] = umap_baseline[:, 1]
df_records["umap_x_ft"] = umap_finetuned[:, 0]
df_records["umap_y_ft"] = umap_finetuned[:, 1]


# --------------------------------------------------------------------------
# 4) Subset single-rhythm records & map to SR, AFIB, SB, GSVT, PACE
# --------------------------------------------------------------------------
def map_ptbxl_codes_to_label(rhythm_codes):
    """
    If there's exactly 1 scp_code in rhythm_codes, try to map it to an
    Arrhythmia-style label using PTBXL_ARR_MAP. Otherwise return None.
    """
    if len(rhythm_codes) == 1:
        code = rhythm_codes[0]
        return PTBXL_ARR_MAP.get(code, None)
    return None


df_records["single_rhythm_label"] = df_records["valid_rhythm_codes"].apply(
    map_ptbxl_codes_to_label
)

# Create a boolean mask for single-rhythm records
mask_single = df_records["single_rhythm_label"].notnull()
df_single = df_records[mask_single].copy()
print(f"Found {len(df_single)} single-rhythm PTB-XL records.")


# --------------------------------------------------------------------------
# 5) Prepare Plot: Baseline vs. Fine-tuned UMAP
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# fig.suptitle(
#     "PTB-XL Only: Embedding Visualization (Baseline vs. Fine-tuned)",
#     fontsize=18,
#     fontweight="bold",
# )


def plot_umap(ax, df, xcol, ycol, title):
    ax.set_title(title, fontsize=14)

    # 1) Plot all records in light gray
    ax.scatter(
        df[xcol],
        df[ycol],
        color="lightgray",
        edgecolor="none",
        s=40,
        alpha=0.6,
        label="All PTB-XL Records",
    )

    # 2) Overlay single-rhythm in color
    for row in df_single.itertuples():
        lbl = getattr(row, "single_rhythm_label")
        xval = getattr(row, xcol)
        yval = getattr(row, ycol)

        # If label is known, pick a color from FIXED_LABEL_COLORS
        # or default to some gray if missing
        color = FIXED_LABEL_COLORS.get(lbl, (0.5, 0.5, 0.5))

        ax.scatter(
            xval,
            yval,
            c=[color],
            marker=DEFAULT_MARKER,
            s=80,
            alpha=0.9,
            edgecolors="white",
            linewidth=0.5,
            # We'll do custom legend below
        )

    ax.set_xlabel("UMAP Dim 1", fontsize=12)
    ax.set_ylabel("UMAP Dim 2", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)


# Left: Baseline
plot_umap(axes[0], df_records, "umap_x_base", "umap_y_base", "Baseline (Pre-trained)")

# Right: Fine-tuned
plot_umap(axes[1], df_records, "umap_x_ft", "umap_y_ft", "Fine-tuned")


# --------------------------------------------------------------------------
# 6) Build Legend
# --------------------------------------------------------------------------
handles = []

# First handle for "All PTB-XL Records"
handles.append(
    Line2D(
        [0],
        [0],
        marker="o",
        color="lightgray",
        label="All PTB-XL Records",
        markersize=10,
        markeredgecolor="none",
        linewidth=0,
    )
)

# Then each known label
labels_present = df_single["single_rhythm_label"].dropna().unique()

# We can group them or simply list them
handles.append(Patch(color="none", label="\nSingle-Rhythm Labels:"))
for lbl in sorted(labels_present):
    color = FIXED_LABEL_COLORS.get(lbl, (0.5, 0.5, 0.5))
    handles.append(
        Line2D(
            [0],
            [0],
            marker=DEFAULT_MARKER,
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=f"  {lbl}",
            linewidth=0,
        )
    )

legend = fig.legend(
    handles=handles,
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=11,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    title="PTB-XL\n(Single-Rhythm)",
    title_fontsize=12,
)

# If needed:
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.savefig("ptbxl_only_embedding_visualization.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done! Saved plot as ptbxl_only_embedding_visualization.png")

# %%
