# %% [markdown]
# # Phase 3 - Cross-Domain Notebook
# In this notebook script, we integrate **Arrhythmia (Chapman)** and **PTB-XL** ECG embeddings
# to analyze cross-domain performance of both the **Baseline** and **Fine-tuned** models.

# %% [markdown]
# ## Imports and Configuration

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

project_root = Path().absolute().parent.parent
sys.path.append(str(project_root))

from src.visualization.embedding_viz import run_umap
from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 12

arr_data = UnifiedDataset(
    project_root / "data", modality=DatasetModality.ECG, dataset_key="arrhythmia"
)
ptbxl_data = UnifiedDataset(
    project_root / "data", modality=DatasetModality.ECG, dataset_key="ptbxl"
)

arr_splits = arr_data.get_splits()
ptbxl_splits = ptbxl_data.get_splits()

arr_test_ids = arr_splits.get("test", [])
ptbxl_test_ids = ptbxl_splits.get("test", [])

arr_md_store = arr_data.metadata_store
ptbxl_md_store = ptbxl_data.metadata_store

pretrained_embedding = "baseline"
finetuned_embedding = "fine_tuned_50"
# %%
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
LABELS_OF_INTEREST.sort()

# Use the same color palette as phase1.py
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

# Use this consistent color map instead of sequential assignment
label2color = FIXED_LABEL_COLORS

arr_records_info = []
emb_base_list_arr = []
emb_ft_list_arr = []

for rid in arr_test_ids:
    meta = arr_md_store.get(rid, {})
    labels_meta = meta.get("labels_metadata", [])

    try:
        emb_base = arr_data.get_embeddings(rid, embeddings_type=pretrained_embedding)
        emb_ft = arr_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
    except Exception:
        # skip if missing embeddings
        continue

    arr_records_info.append({"record_id": rid, "labels_meta": labels_meta})
    emb_base_list_arr.append(emb_base)
    emb_ft_list_arr.append(emb_ft)

df_arr = pd.DataFrame(arr_records_info)
df_arr["row_idx"] = df_arr.index

baseline_arr = np.vstack(emb_base_list_arr)  # shape: (N_arr, emb_dim)
finetuned_arr = np.vstack(emb_ft_list_arr)  # shape: (N_arr, emb_dim)


def find_rhythm_labels(labels_meta):
    """
    Return a list of integration_names that are group="Rhythm"
    *or* 'PACE' (forced override).
    """
    rhythm_names = []
    for lm in labels_meta:
        integration_name = lm.get("integration_name", "")
        group = lm.get("group", "")
        # If group is Rhythm or if it's 'PACE', treat it as Rhythm
        if group == "Rhythm" or integration_name == "PACE":
            rhythm_names.append(integration_name)
    return rhythm_names


df_arr["rhythm_labels"] = df_arr["labels_meta"].apply(find_rhythm_labels)
df_arr["n_rhythm_labels"] = df_arr["rhythm_labels"].apply(len)


def get_single_label_of_interest(row):
    """
    If exactly 1 rhythm label is present, and it's in LABELS_OF_INTEREST,
    return that label.
    """
    if row["n_rhythm_labels"] == 1:
        lbl = row["rhythm_labels"][0]
        if lbl in LABELS_OF_INTEREST:
            return lbl
    return None


df_arr["single_rhythm_label"] = df_arr.apply(get_single_label_of_interest, axis=1)

ptbxl_records_info = []
emb_base_list_ptb = []
emb_ft_list_ptb = []

for rid in ptbxl_test_ids:
    meta = ptbxl_md_store.get(rid, {})
    scp_codes = meta.get("scp_codes", {})
    scp_statements = meta.get("scp_statements", {})

    # Identify rhythm labels
    rhythm_codes = []
    other_codes = []
    
    for code_key in scp_codes:
        code_info = scp_statements.get(code_key, {})
        # Check if it's a rhythm code
        if code_info.get("rhythm", 0.0) == 1.0 and code_key in PTBXL_ARR_MAP:
            rhythm_codes.append(code_key)
        # Check if it's a diagnostic or form code
        elif code_info.get("diagnostic", 0.0) == 1.0 or code_info.get("form", 0.0) == 1.0:
            other_codes.append(code_key)
    
    if not rhythm_codes:
        continue
    
    # we take the one with higher confidence
    primary_rhythm_code = rhythm_codes[0]
    mapped_rhythm_label = PTBXL_ARR_MAP[primary_rhythm_code]
    
    # Determine if this is a single-labeled record (only rhythm, no other diagnostics or NORM)
    is_single_labeled = len(other_codes) == 0 or (len(other_codes) == 1 and other_codes[0] == "NORM")
    
    try:
        emb_base = ptbxl_data.get_embeddings(rid, embeddings_type=pretrained_embedding)
        emb_ft = ptbxl_data.get_embeddings(rid, embeddings_type=finetuned_embedding)
    except Exception:
        continue
    
    # Store additional metadata
    age = meta.get("age", np.nan)
    sex = meta.get("sex", np.nan)
    site = meta.get("site", np.nan)
    dev = meta.get("device", "NA")
    weight = meta.get("weight", np.nan)
    height = meta.get("height", np.nan)

    ptbxl_records_info.append({
        "record_id": rid,
        "rhythm_codes": rhythm_codes,
        "other_codes": other_codes,
        "rhythm_label": mapped_rhythm_label,
        "is_single_labeled": is_single_labeled,
        "age": age,
        "sex": sex,
        "site": site,
        "device": dev,
        "weight": weight,
        "height": height,
    })
    
    emb_base_list_ptb.append(emb_base)
    emb_ft_list_ptb.append(emb_ft)

df_ptbxl = pd.DataFrame(ptbxl_records_info)
df_ptbxl["row_idx"] = df_ptbxl.index

# Create a column for other_label (for dual-labeled records)
def get_other_label(row):
    """Get a concatenated string of other diagnostic/form labels"""
    if row["is_single_labeled"]:
        return None
    else:
        return "+".join(row["other_codes"])

df_ptbxl["other_label"] = df_ptbxl.apply(get_other_label, axis=1)

baseline_ptb = np.vstack(emb_base_list_ptb)
finetuned_ptb = np.vstack(emb_ft_list_ptb)


umap_params = dict(n_neighbors=30, n_components=2, min_dist=0.25, metric="euclidean", random_state=42)


def combine_and_umap(arr_emb, ptb_emb):
    """
    Stack arr_emb + ptb_emb, run UMAP, split results back.
    """
    # import numpy as np

    combined = np.vstack([arr_emb, ptb_emb])
    combined_umap = run_umap(combined, **umap_params)
    coords_arr = combined_umap[: len(arr_emb)]
    coords_ptb = combined_umap[len(arr_emb) :]
    return coords_arr, coords_ptb


arr_baseline_umap, ptb_baseline_umap = combine_and_umap(baseline_arr, baseline_ptb)
arr_finetuned_umap, ptb_finetuned_umap = combine_and_umap(finetuned_arr, finetuned_ptb)

df_arr_baseline = df_arr.copy()
df_arr_baseline["umap_x"] = arr_baseline_umap[:, 0]
df_arr_baseline["umap_y"] = arr_baseline_umap[:, 1]

df_arr_finetuned = df_arr.copy()
df_arr_finetuned["umap_x"] = arr_finetuned_umap[:, 0]
df_arr_finetuned["umap_y"] = arr_finetuned_umap[:, 1]

df_ptbxl_baseline = df_ptbxl.copy()
df_ptbxl_baseline["umap_x"] = ptb_baseline_umap[:, 0]
df_ptbxl_baseline["umap_y"] = ptb_baseline_umap[:, 1]

df_ptbxl_finetuned = df_ptbxl.copy()
df_ptbxl_finetuned["umap_x"] = ptb_finetuned_umap[:, 0]
df_ptbxl_finetuned["umap_y"] = ptb_finetuned_umap[:, 1]

# %%

# just used to make kde plots not explode
def remove_outliers_2d(points, z_thresh=3.0):
    if len(points) == 0:
        return points
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std[std < 1e-12] = 1e-12  # avoid divide-by-zero

    z_scores = (points - mean) / std
    dist = np.sqrt((z_scores**2).sum(axis=1))
    return points[dist <= z_thresh]


# Create a function to plot joint plots with marginal KDEs
def create_joint_plot(df_arr, df_ptb, title, figsize=(12, 8)):
    arr_data = df_arr[["umap_x", "umap_y"]].copy()
    arr_data["dataset"] = "Chapman"

    ptb_data = df_ptb[["umap_x", "umap_y"]].copy()
    ptb_data["dataset"] = "PTB-XL"

    combined_data = pd.concat([arr_data, ptb_data])

    g = sns.JointGrid(
        data=combined_data, x="umap_x", y="umap_y", height=figsize[0], ratio=4
    )

    g.ax_joint.scatter(
        df_arr["umap_x"],
        df_arr["umap_y"],
        color="lightgray",
        s=20,
        alpha=0.6,
        label="Chapman (all)",
    )

    for lbl in LABELS_OF_INTEREST:
        sub = df_arr.loc[
            df_arr["single_rhythm_label"] == lbl, ["umap_x", "umap_y"]
        ].values

        if len(sub) < 5:
            continue

        if len(sub) < 5:
            continue

        sub_df = pd.DataFrame(sub, columns=["umap_x", "umap_y"])
        sns.kdeplot(
            data=sub_df,
            x="umap_x",
            y="umap_y",
            fill=True,
            levels=4,
            alpha=0.25,
            color=label2color.get(lbl, "red"),
            ax=g.ax_joint,
            label=None,
        )

    g.ax_joint.scatter(
        df_ptb["umap_x"],
        df_ptb["umap_y"],
        c=[label2color.get(m, "black") for m in df_ptb["rhythm_label"]],
        marker="*",
        s=120,
        edgecolors="white",
        linewidth=0.6,
        alpha=0.9,
        label="PTB-XL (single-rhythm, overlay)",
    )

    sns.kdeplot(
        data=arr_data, x="umap_x", color="lightblue", ax=g.ax_marg_x, label="Chapman"
    )
    sns.kdeplot(
        data=ptb_data, x="umap_x", color="orange", ax=g.ax_marg_x, label="PTB-XL"
    )
    sns.kdeplot(data=arr_data, y="umap_y", color="lightblue", ax=g.ax_marg_y)
    sns.kdeplot(data=ptb_data, y="umap_y", color="orange", ax=g.ax_marg_y)

    g.ax_marg_x.legend(fontsize=12)

    g.ax_joint.set_xlabel("UMAP Dim 1")
    g.ax_joint.set_ylabel("UMAP Dim 2")
    g.ax_joint.set_title(title, fontsize=14)
    g.ax_joint.grid(True, linestyle="--", alpha=0.5)

    return g


fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.3)

g1 = create_joint_plot(
    df_arr_baseline, df_ptbxl_baseline, f"Baseline (Pre-trained) Embeddings\n PTB-XL and Chapman"
)
g2 = create_joint_plot(
    df_arr_finetuned, df_ptbxl_finetuned, f"Fine-tuned (Chapman) Embeddings\n PTB-XL and Chapman"
)

g1.savefig("baseline_joint_plot.png")
g2.savefig("finetuned_joint_plot.png")


def plot_density_contours(
    ax, df_arr_plot, df_ptb_plot, panel_title, ignore_outliers=False
):
    ax.set_title(panel_title, fontsize=14)
    ax.scatter(
        df_arr_plot["umap_x"],
        df_arr_plot["umap_y"],
        color="lightgray",
        edgecolor="none",
        s=40,
        alpha=0.6,
        label="Chapman (all)",
    )

    for lbl in LABELS_OF_INTEREST:
        sub = df_arr_plot.loc[
            df_arr_plot["single_rhythm_label"] == lbl, ["umap_x", "umap_y"]
        ].values

        if len(sub) < 5:
            continue

        if ignore_outliers:
            sub = remove_outliers_2d(sub, z_thresh=3.0 if lbl != "PACE" else 1.0)

        if len(sub) < 5:
            continue

        sub_df = pd.DataFrame(sub, columns=["umap_x", "umap_y"])
        sns.kdeplot(
            data=sub_df,
            x="umap_x",
            y="umap_y",
            fill=True,
            levels=4,
            alpha=0.25,
            color=label2color.get(lbl, "red"),
            ax=ax,
            label=None,
        )

    # Plot PTB-XL single-labeled records with circles
    single_mask = df_ptb_plot["is_single_labeled"]
    if single_mask.any():
        # Group by rhythm_label and plot each with its own marker
        for lbl in df_ptb_plot.loc[single_mask, "rhythm_label"].unique():
            label_mask = single_mask & (df_ptb_plot["rhythm_label"] == lbl)
            ax.scatter(
                df_ptb_plot.loc[label_mask, "umap_x"],
                df_ptb_plot.loc[label_mask, "umap_y"],
                c=label2color.get(lbl, "black"),
                marker=FIXED_LABEL_MARKERS.get(lbl, "o"),
                s=80,
                edgecolors="white",
                linewidth=0.6,
                alpha=0.9,
            label="PTB-XL (single-rhythm)",
            )
    
    # Plot PTB-XL dual-labeled records with triangles
    # dual_mask = ~df_ptb_plot["is_single_labeled"]
    # if dual_mask.any():
    #     ax.scatter(
    #         df_ptb_plot.loc[dual_mask, "umap_x"],
    #         df_ptb_plot.loc[dual_mask, "umap_y"],
    #         c=[label2color.get(m, "black") for m in df_ptb_plot.loc[dual_mask, "rhythm_label"]],
    #         marker="^",
    #         s=80,
    #         edgecolors="white",
    #         linewidth=0.6,
    #         alpha=0.9,
    #         label="PTB-XL (multi-labeled)",
    #     )

    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, linestyle="--", alpha=0.5)

    return ax


plot_density_contours(
    axes[0],
    df_arr_baseline,
    df_ptbxl_baseline,
    f"Baseline (Pre-trained) Embeddings\n PTB-XL and Chapman",
)

plot_density_contours(
    axes[1], df_arr_finetuned, df_ptbxl_finetuned, f"Fine-Tuned Embeddings\n PTB-XL and Chapman"
)

plt.tight_layout(rect=[0, 0.2, 1, 0.97])  # Make room for the suptitle

general_handles = [
    Line2D([0], [0], marker="o", color="lightgray", markersize=8, linewidth=0),
    # Line2D([0], [0], marker="*", color="black", markersize=15, linewidth=0),
]
general_labels = ["Chapman (all)", "PTB-XL (single-rhythm, overlay)"]

contour_handles = [Patch(color="none")]
contour_labels = ["Chapman Single-Label (Rhythm) Contours:"]

# Use LABELS_OF_INTEREST order for consistent sorting
for lbl in LABELS_OF_INTEREST:
    clr = label2color.get(lbl, "black")
    contour_handles.append(Patch(facecolor=clr, edgecolor="none", alpha=0.25))
    contour_labels.append(f"{lbl} (Chapman)")

ptbxl_handles = [Patch(color="none")]
ptbxl_labels = ["PTB-XL Single-Rhythm (Mapped Labels):"]

# Use LABELS_OF_INTEREST order for consistent sorting
# Only include labels that are actually present in the data
ptbxl_labels_present = set(df_ptbxl["rhythm_label"].unique())
for lbl in LABELS_OF_INTEREST:
    if lbl in ptbxl_labels_present:
        clr = label2color.get(lbl, "black")
        ptbxl_handles.append(
            Line2D([0], [0], marker=FIXED_LABEL_MARKERS.get(lbl, "o"), color=clr, markersize=8, linewidth=0)
        )
        ptbxl_labels.append(f"{lbl}")

legend1 = fig.legend(
    general_handles,
    general_labels,
    loc="lower left",
    bbox_to_anchor=(0.2, -0.04),
    frameon=False,
    fontsize=10,
    handletextpad=0.5,
)

legend2 = fig.legend(
    contour_handles,
    contour_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    frameon=False,
    fontsize=10,
    handletextpad=0.5,
)

legend3 = fig.legend(
    ptbxl_handles,
    ptbxl_labels,
    loc="lower right",
    bbox_to_anchor=(0.83, -0.14),
    frameon=False,
    fontsize=10,
    handletextpad=0.5,
)

legend_frame = plt.Rectangle(
    (0.15, -0.14),
    0.7,
    0.18,
    transform=fig.transFigure,
    fill=False,
    edgecolor="gray",
    linewidth=0.5,
)
fig.patches.append(legend_frame)

plt.tight_layout(rect=[0, 0.15, 1, 0.97])
fig.savefig("results/cross_domain_embedding_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
#%%

def create_faceted_joint_plots(df_arr, df_ptb, title, figsize=(12, 12)):
    plt.style.use("seaborn-v0_8-whitegrid")
    
    n_labels = len(LABELS_OF_INTEREST)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten in case there's more than one row
    if n_rows == 1:
        axes = np.array(axes)  # ensure it's an array
    axes = axes.flatten()

    # Calculate global min/max for consistent axes across all facets
    all_x_values = pd.concat([df_arr['umap_x'], df_ptb['umap_x']])
    all_y_values = pd.concat([df_arr['umap_y'], df_ptb['umap_y']])
    
    # Use percentiles to avoid extreme outliers affecting the scale
    x_min, x_max = np.percentile(all_x_values, [1, 99])
    y_min, y_max = np.percentile(all_y_values, [1, 99])
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1

    for i, lbl in enumerate(LABELS_OF_INTEREST):
        if i >= len(axes):
            break

        ax = axes[i]

        # Get all Chapman points and Chapman points for this specific label
        arr_all = df_arr.copy()  # All Chapman points
        arr_lbl = df_arr[df_arr["single_rhythm_label"] == lbl].copy()  # Chapman points for this label
        
        # Split PTB-XL points into single-labeled and multilabeled
        ptb_single = df_ptb[(df_ptb["rhythm_label"] == lbl) & (df_ptb["is_single_labeled"])].copy()
        ptb_dual = df_ptb[(df_ptb["rhythm_label"] == lbl) & (~df_ptb["is_single_labeled"])].copy()
        
        # Combine for total count check
        ptb_lbl = pd.concat([ptb_single, ptb_dual])

        if len(arr_lbl) < 5 or len(ptb_lbl) < 2:
            ax.text(
                0.5,
                0.5,
                f"Not enough data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
            
        if len(ptb_single) > 100:
            ptb_single = ptb_single.sample(100, random_state=42)
        if len(ptb_dual) > 100:
            ptb_dual = ptb_dual.sample(100, random_state=42)
        # Update ptb_lbl after sampling
        ptb_lbl = pd.concat([ptb_single, ptb_dual])

        # Plot ALL Chapman points in grey
        scatter_all = sns.scatterplot(
            data=arr_all,
            x="umap_x",
            y="umap_y",
            color="lightgray",  # Always gray for all Chapman points
            s=20,
            alpha=0.3,  # Lower alpha for all points
            ax=ax,
            label="Chapman (all)",
            )
        
        # Plot PTB-XL single-labeled points for this label with the appropriate marker for this label
        if len(ptb_single) > 0:
            scatter_ptb_single = sns.scatterplot(
                data=ptb_single,
                x="umap_x",
                y="umap_y",
                color=label2color.get(lbl, "black"),
                marker=FIXED_LABEL_MARKERS.get(lbl, "o"),
                s=90,
                edgecolors="white",
                linewidth=0.6,
                alpha=0.8,
                ax=ax,
                label="PTB-XL",
                )
        
        # Plot PTB-XL multilabeled points for this label with triangle marker
        if len(ptb_dual) > 0:
            scatter_ptb_dual = sns.scatterplot(
                data=ptb_dual,
                x="umap_x",
                y="umap_y",
                color=label2color.get(lbl, "black"),
                marker="^",
                s=80,
                edgecolors="white",
                linewidth=0.6,
                alpha=0.8,
                ax=ax,
                label="PTB-XL+",
                )
        
        # KDE contour for Chapman points for this label in the LABEL COLOR
        if len(arr_lbl) >= 5:
            sns.kdeplot(
                data=arr_lbl,
                x="umap_x",
                y="umap_y",
                fill=True,
                levels=4,
                alpha=0.25,
                color=label2color.get(lbl, "black"),  # Use label color for contours
                ax=ax,
                label=f"{lbl} Contour",  # Add specific contour label for the legend
            )

        # Set consistent axis limits for all facets
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
        ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")

        divider = make_axes_locatable(ax)

        # Top marginal density plot
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )

        # Set the x-limits to match the main plot
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_title(f"Label: {lbl}", fontsize=14, pad=5)
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        
        # Plot all three density curves in the top margin
        if len(arr_all) >= 5:
            sns.kdeplot(data=arr_all, x="umap_x", color="lightgray", ax=ax_top, label="Chapman (all)")
        if len(arr_lbl) >= 5:
            sns.kdeplot(data=arr_lbl, x="umap_x", color=label2color.get(lbl, "black"), 
                       linestyle="--", ax=ax_top, label="Chapman")
        if len(ptb_lbl) >= 2:
            sns.kdeplot(
                data=ptb_lbl, x="umap_x", color=label2color.get(lbl, "black"), 
                ax=ax_top, label="PTB-XL"
            )
        
        # Add a small legend to the top margin plot with a cleaner style
        if len(arr_all) >= 5 and len(arr_lbl) >= 5 and len(ptb_lbl) >= 2:
            ax_top.legend(["Chapman (all)", "Chapman", "PTB-XL"], fontsize=8)
        
        # Right marginal density plot
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False
        )
        
        # Set the y-limits to match the main plot
        ax_right.set_ylim(ax.get_ylim())
        
        # Plot all three density curves in the right margin
        if len(arr_all) >= 5:
            sns.kdeplot(data=arr_all, y="umap_y", color="lightgray", ax=ax_right)
        if len(arr_lbl) >= 5:
            sns.kdeplot(data=arr_lbl, y="umap_y", color=label2color.get(lbl, "black"), 
                       linestyle="--", ax=ax_right)
        if len(ptb_lbl) >= 2:
            sns.kdeplot(
                data=ptb_lbl,
                y="umap_y",
                color=label2color.get(lbl, "black"),
                ax=ax_right,
            )
        
        # Add legend to the main plot with explicit contour inclusion
        # Create custom legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor="lightgray", 
                   markersize=8, label='Chapman (all)'),
        ]
        
        # Get the original counts (before sampling)
        original_single_count = len(df_ptb[(df_ptb["rhythm_label"] == lbl) & (df_ptb["is_single_labeled"])])
        original_dual_count = len(df_ptb[(df_ptb["rhythm_label"] == lbl) & (~df_ptb["is_single_labeled"])])
        
        if len(ptb_single) > 0:
            legend_elements.append(
                Line2D([0], [0], marker=FIXED_LABEL_MARKERS.get(lbl, "o"), color='none', markerfacecolor=label2color.get(lbl, "black"), 
                       markeredgecolor=label2color.get(lbl, "black"), markersize=8, label=f'PTB-XL (n={original_single_count})')
            )
            
        if len(ptb_dual) > 0:
            legend_elements.append(
                Line2D([0], [0], marker='^', color='none', markerfacecolor=label2color.get(lbl, "black"), 
                       markeredgecolor=label2color.get(lbl, "black"), markersize=8, label=f'PTB-XL+ (n={original_dual_count})')
            )
            
        if len(arr_lbl) >= 5:
            legend_elements.append(
                Patch(facecolor=label2color.get(lbl, "black"), alpha=0.25, 
                  label=f'Chapman Contour')
            )
        
        # Add the custom legend to each facet
        ax.legend(handles=legend_elements, fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


from mpl_toolkits.axes_grid1 import make_axes_locatable

create_faceted_joint_plots(
    df_arr_baseline,
    df_ptbxl_baseline,
    "Baseline (Pre-trained) Embeddings on Chapman (Faceted)",
    figsize=(12, 12),
)

create_faceted_joint_plots(
    df_arr_finetuned,
    df_ptbxl_finetuned,
    "Fine-Tuned (Chapman) Embeddings (Faceted)",
    figsize=(12, 12),
)

plt.savefig("results/baseline_faceted_joint_plots.png", dpi=150, bbox_inches="tight")
plt.savefig("results/finetuned_faceted_joint_plots.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Faceted Joint Plots with PTB-XL Metadata Coloring
# Now we extend the facet logic to color PTB-XL records by a chosen metadata column,
# while keeping the Arrhythmia (Chapman) points in gray with KDE contours.

# %%

def create_faceted_joint_plots_with_metadata(
    df_arr,
    df_ptb,
    metadata_col,
    title="Faceted Joint Plots by Label",
    label_column_arr="single_rhythm_label",   # label col in Arr df
    label_column_ptb="rhythm_label",          # label col in PTB df
    labels_of_interest=None,                  # e.g. ["AFIB", "GSVT", ...]
    continuous=False,                         # True if metadata is numeric/continuous
    z_thresh_arr=3.0,
    z_thresh_ptb=3.0,
    figsize=(12, 12),
    cmap_name="viridis"
):
    """
    Create facet plots for each label of interest.
    - The Arr (Chapman) data is plotted in gray (plus optional KDE).
    - The PTB-XL points are color-coded by 'metadata_col'.

    Parameters
    ----------
    df_arr : pd.DataFrame
        Must contain columns: ['umap_x', 'umap_y', label_column_arr, ...]
    df_ptb : pd.DataFrame
        Must contain columns: ['umap_x', 'umap_y', label_column_ptb, metadata_col, ...]
    metadata_col : str
        Which PTB-XL metadata field to color by. E.g. 'nurse', 'sex', 'device', etc.
    label_column_arr : str
        Column in df_arr that identifies the single-labeled class (e.g. "single_rhythm_label").
    label_column_ptb : str
        Column in df_ptb that identifies the rhythm label (e.g. "rhythm_label").
    labels_of_interest : list
        Which labels to facet over (e.g. ["AFIB", "GSVT", "SB", ...]).
    continuous : bool
        If True, we treat metadata_col as numeric and use a continuous color scale.
        If False, we treat it as categorical.
    z_thresh_arr : float
        Z-score cutoff for removing outliers in Arr data (for KDE).
    z_thresh_ptb : float
        Z-score cutoff for removing outliers in PTB data (if you want to remove them).
    figsize : tuple
        Figure size.
    cmap_name : str
        Matplotlib colormap name if continuous = True.

    Returns
    -------
    fig : matplotlib Figure
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    
    labels_of_interest = LABELS_OF_INTEREST
    


    # Prepare subplots
    n_labels = len(labels_of_interest)
    n_cols = 3  # or 2, if you prefer fewer columns
    n_rows = 2  # Force 2 rows to fill all 6 slots

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten in case there's more than one row
    if n_rows == 1:
        axes = np.array(axes)  # ensure it's an array
    axes = axes.flatten()

    # For continuous data, set up a color normalization
    # We'll gather all the valid numeric values across all labels
    if continuous:
        # For numeric/continuous data
        valid_meta = df_ptb[metadata_col].dropna()
        # Cap age at 90 for visualization purposes
        if metadata_col.lower() == 'age':
            df_ptb[metadata_col] = df_ptb[metadata_col].clip(lower=0, upper=90)
            valid_meta = valid_meta.clip(lower=0, upper=90)
            print(f"Age capped at 90 (original highest: {valid_meta.max()})")

        if len(valid_meta) > 0:
            vmin = valid_meta.min()
            vmax = valid_meta.max()
        else:
            vmin, vmax = 0, 1  # fallback
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

    for i, lbl in enumerate(labels_of_interest):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract Arr & PTB rows that match the label
        arr_all = df_arr.copy()
        arr_lbl = df_arr[df_arr[label_column_arr] == lbl].copy()
        
        # Split PTB-XL data into single-labeled and multi-labeled
        ptb_single = df_ptb[(df_ptb[label_column_ptb] == lbl) & (df_ptb["is_single_labeled"])].copy()
        ptb_dual = df_ptb[(df_ptb[label_column_ptb] == lbl) & (~df_ptb["is_single_labeled"])].copy()
        
        # Combine for total count checks
        ptb_lbl = pd.concat([ptb_single, ptb_dual])

        # If there's not enough data, skip plotting
        if len(arr_lbl) < 3 or (len(ptb_single) < 1 and len(ptb_dual) < 1):
            ax.text(
                0.5, 0.5,
                "Not enough data",
                ha="center", va="center",
                transform=ax.transAxes
            )
            ax.set_title(f"Label: {lbl}", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot ALL Chapman points in grey
        scatter_all = sns.scatterplot(
            data=arr_all,
            x="umap_x",
            y="umap_y",
            color="lightgray",  # Always gray for all Chapman points
            s=20,
            alpha=0.3,  # Lower alpha for all points
            ax=ax,
            label="Chapman (all)",
        )
        
        # Now color the PTB points by metadata_col
        if continuous:
            # For numeric/continuous data
            if len(ptb_single) > 0:
                cvals_single = []
                for idx2, row2 in ptb_single.iterrows():
                    val = row2[metadata_col]
                    # If it's NaN, we can skip or set a default color
                    if pd.isna(val):
                        cvals_single.append("black")
                    else:
                        cvals_single.append(cmap(norm(val)))
                ax.scatter(
                    ptb_single["umap_x"],
                    ptb_single["umap_y"],
                    c=cvals_single,
                    marker=FIXED_LABEL_MARKERS.get(lbl, "o"),
                    s=90,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"PTB-XL (n={len(ptb_single)})"
                )
            
            if len(ptb_dual) > 0:
                cvals_dual = []
                for idx2, row2 in ptb_dual.iterrows():
                    val = row2[metadata_col]
                    # If it's NaN, we can skip or set a default color
                    if pd.isna(val):
                        cvals_dual.append("black")
                    else:
                        cvals_dual.append(cmap(norm(val)))
                ax.scatter(
                    ptb_dual["umap_x"],
                    ptb_dual["umap_y"],
                    c=cvals_dual,
                    marker="^",
                    s=80,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"PTB-XL+ (n={len(ptb_dual)})"
                )
        else:
            # For categorical data
            if len(ptb_single) > 0 or len(ptb_dual) > 0:
                # Get all unique values for this metadata column across the entire dataset
                all_unique_vals = df_ptb[metadata_col].dropna().unique()
                all_unique_vals = sorted(all_unique_vals, key=lambda x: str(x))
                
                # Pick a color palette with enough distinct colors
                palette = sns.color_palette("tab10", n_colors=len(all_unique_vals))
                cat2color = dict(zip(all_unique_vals, palette))

                # We'll plot each category separately for single-labeled
                if len(ptb_single) > 0:
                    for cat in all_unique_vals:
                        subcat = ptb_single[ptb_single[metadata_col] == cat]
                        if len(subcat) > 0:
                            ax.scatter(
                                subcat["umap_x"],
                                subcat["umap_y"],
                                color=cat2color[cat],
                                marker=FIXED_LABEL_MARKERS.get(lbl, "o"),
                                s=90,
                                alpha=0.8,
                                edgecolor="white",
                                linewidth=0.5,
                                label=f"{cat}"
                            )
                
                # We'll plot each category separately for multi-labeled
                if len(ptb_dual) > 0:
                    for cat in all_unique_vals:
                        subcat = ptb_dual[ptb_dual[metadata_col] == cat]
                        if len(subcat) > 0:
                            ax.scatter(
                                subcat["umap_x"],
                                subcat["umap_y"],
                                color=cat2color[cat],
                                marker="^",
                                s=80,
                                alpha=0.8,
                                edgecolor="white",
                                linewidth=0.5,
                                label=f"{cat}+"
                            )

        # Title and axis labels
        # Remove title from main plot since we'll put it on the top margin
        ax.set_title("")  
        ax.set_xlabel("UMAP Dim 1" if i >= len(axes) - n_cols else "")
        ax.set_ylabel("UMAP Dim 2" if i % n_cols == 0 else "")

        # Make top/bottom marginal plots: replicate your existing approach
        divider = make_axes_locatable(ax)

        # Top marginal density plot
        ax_top = divider.append_axes("top", size="15%", pad=0.1)
        ax_top.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

        # Set the x-limits to match the main plot
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_title(f"Label: {lbl}", fontsize=14, pad=5)
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        
        # Plot density curves in the top margin
        if len(arr_all) >= 5:
            sns.kdeplot(data=arr_all, x="umap_x", color="lightgray", ax=ax_top, label="Chapman (all)")
        
        # Combine PTB data for KDE plot
        ptb_lbl = pd.concat([ptb_single, ptb_dual])
        if len(ptb_lbl) >= 2:
            sns.kdeplot(
                data=ptb_lbl, x="umap_x", color=label2color.get(lbl, "black"), 
                ax=ax_top, label="PTB-XL"
            )

        ax_top.set_title(f"Label: {lbl}", fontsize=14, pad=5)  # Add label title here
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        
        # Add a small legend to the top margin plot with a cleaner style
        if len(arr_all) >= 5 and len(ptb_lbl) >= 2:
            ax_top.legend(["Chapman (all)", "PTB-XL"], fontsize=8)

        # Right marginal density plot
        ax_right = divider.append_axes("right", size="15%", pad=0.1)
        ax_right.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

        if len(arr_all) >= 5:
            sns.kdeplot(data=arr_all, y="umap_y", color="lightgray", ax=ax_right)
        if len(ptb_lbl) >= 2:
            sns.kdeplot(data=ptb_lbl, y="umap_y", color=label2color.get(lbl, "black"), ax=ax_right)

        ax_right.set_xlabel("")
        ax_right.set_xticks([])
        
        # Add legend to the main plot
        # Create custom legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor="lightgray", 
                   markersize=8, label='Chapman (all)'),
        ]
        
        # Get the original counts (before sampling)
        original_single_count = len(df_ptb[(df_ptb[label_column_ptb] == lbl) & (df_ptb["is_single_labeled"])])
        original_dual_count = len(df_ptb[(df_ptb[label_column_ptb] == lbl) & (~df_ptb["is_single_labeled"])])
        
        if len(ptb_single) > 0:
            legend_elements.append(
                Line2D([0], [0], marker=FIXED_LABEL_MARKERS.get(lbl, "o"), color='none', markerfacecolor=label2color.get(lbl, "black"), 
                       markeredgecolor=label2color.get(lbl, "black"), markersize=8, label=f'PTB-XL (n={original_single_count})')
            )
            
        if len(ptb_dual) > 0:
            legend_elements.append(
                Line2D([0], [0], marker='^', color='none', markerfacecolor=label2color.get(lbl, "black"), 
                       markeredgecolor=label2color.get(lbl, "black"), markersize=8, label=f'PTB-XL+ (n={original_dual_count})')
            )
        
        # Add the custom legend to each facet
        ax.legend(handles=legend_elements, fontsize=8, loc='best')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Overall figure title
    # fig.suptitle(title, fontsize=16)

    # Possibly create one overall color legend or colorbar for the entire figure
    if continuous:
        # Add one colorbar on the right side of the figure
        fig.subplots_adjust(right=0.70)  # Make much more space on the right
        cbar_ax = fig.add_axes([1.03, 0.15, 0.02, 0.7])  # x, y, width, height - moved even further right
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(metadata_col)
    else:
        # For categorical, we can build a single legend, or rely on each axis's local legend.
        # Move the legend to the bottom right empty space
        if metadata_col != label_column_ptb:  # Only add the global legend for non-label metadata
            # Get all unique values across all PTB-XL data
            all_unique_vals = df_ptb[metadata_col].dropna().unique()
            all_unique_vals = sorted(all_unique_vals, key=lambda x: str(x))
            
            # Create a palette for all possible values
            all_palette = sns.color_palette("tab10", n_colors=len(all_unique_vals))
            all_cat2color = dict(zip(all_unique_vals, all_palette))
            
            # Create legend handles for all values
            all_handles = []
            for val in all_unique_vals:
                # For single-labeled
                all_handles.append(
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=all_cat2color[val],
                           markeredgecolor=all_cat2color[val], markersize=8, label=f'{val}')
                )
            
            # Position the legend in the bottom right corner
            # Create a new axis in the bottom right corner for the legend
            
            # Standardize the vertical position for all legend types
            # Calculate appropriate dimensions based on legend size
            ncol = 2 if len(all_unique_vals) > 10 else 1
            
            # Calculate legend height based on number of items
            if len(all_unique_vals) <= 5:  # Small legend (like 'sex')
                legend_height = 0.15
                legend_width = 0.15
                fontsize = 10
                bottom_position = 0.25  # Standard bottom position for all legends
            else:  # Large legend (like 'site' or 'device')
                legend_height = min(0.40, max(0.25, 0.015 * len(all_unique_vals)))
                legend_width = 0.28 if ncol == 2 else 0.20
                fontsize = 9
                bottom_position = 0.25  # Same standard position for consistency
            
            # Create the legend axis with standardized positioning
            legend_ax = fig.add_axes([0.73, bottom_position, legend_width, legend_height])
            legend_ax.axis('off')
            
            # Create the legend with appropriate settings based on size
            legend = legend_ax.legend(
                handles=all_handles,
                labels=all_unique_vals,
                loc='center',
                frameon=True,
                title=metadata_col,
                fontsize=fontsize,
                ncol=ncol,
                labelspacing=0.3
            )
            legend.set_title(metadata_col, prop={'size': 12, 'weight': 'bold'})
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# %% [markdown]
# ## Example Usage of Metadata-Colored Facet Plots

# %%
# Example 1: Create faceted plots colored by age (continuous)
fig_age = create_faceted_joint_plots_with_metadata(
    df_arr_finetuned,
    df_ptbxl_finetuned,
    metadata_col="age",
    title="Faceted Plots by Label - Colored by Age (Fine-tuned)",
    labels_of_interest=LABELS_OF_INTEREST,
    continuous=True,
    cmap_name="plasma",
)
plt.show()

# %%
# Example 2: Create faceted plots colored by sex (categorical)
fig_sex = create_faceted_joint_plots_with_metadata(
    df_arr_finetuned,
    df_ptbxl_finetuned,
    metadata_col="sex",
    title="Faceted Plots by Label - Colored by Sex (Fine-tuned)",
    labels_of_interest=LABELS_OF_INTEREST,
    continuous=False,
)
plt.savefig("finetuned_faceted_by_sex.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Example 3: Create faceted plots colored by site (categorical)
fig_site = create_faceted_joint_plots_with_metadata(
    df_arr_finetuned,
    df_ptbxl_finetuned,
    metadata_col="site",
    title="Faceted Plots by Label - Colored by Site (Fine-tuned)",
    labels_of_interest=LABELS_OF_INTEREST,
    continuous=False,
)
plt.savefig("finetuned_faceted_by_site.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Example 4: Create faceted plots colored by device (categorical)
fig_device = create_faceted_joint_plots_with_metadata(
    df_arr_finetuned,
    df_ptbxl_finetuned,
    metadata_col="device",
    title="Faceted Plots by Label - Colored by Device (Fine-tuned)",
    labels_of_interest=LABELS_OF_INTEREST,
    continuous=False,
)
plt.savefig("finetuned_faceted_by_device.png", dpi=150, bbox_inches="tight")
plt.show()
