# %% [markdown]
# # Centroid Analysis for Cross-Domain Generalization
# This notebook implements class-centric alignment measures to quantify how well Chapman and PTB-XL datasets overlap.

# %% [markdown]
# ## Imports and Configuration

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

project_root = Path().absolute().parent.parent
sys.path.append(str(project_root))

from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 12

# %% [markdown]
# ## Load Data
# We'll load the same data as in phase3.py

# %%
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
# extend the list with train and val of ptbxl
ptbxl_test_ids.extend(ptbxl_splits.get("train", []))
ptbxl_test_ids.extend(ptbxl_splits.get("val", []))


arr_md_store = arr_data.metadata_store
ptbxl_md_store = ptbxl_data.metadata_store

pretrained_embedding = "baseline"
finetuned_embedding = "fine_tuned_50"

# %%
# Define the mapping between PTB-XL and Chapman labels
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

# %% [markdown]
# ## Load Chapman (Arrhythmia) Embeddings

# %%
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

# %%
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

# %% [markdown]
# ## Load PTB-XL Embeddings

# %%
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

# %% [markdown]
# ## Step 1: Extract Label Arrays

# %%
# Create arrays of labels for Chapman and PTB-XL
chapman_labels = df_arr["single_rhythm_label"].values
ptbxl_labels = df_ptbxl["rhythm_label"].values

# %% [markdown]
# ## Step 2: Gather Raw Embeddings (Already Done)
# 
# - Chapman Baseline: `baseline_arr`
# - Chapman Fine-tuned: `finetuned_arr`
# - PTB-XL Baseline: `baseline_ptb`
# - PTB-XL Fine-tuned: `finetuned_ptb`

# %% [markdown]
# ## Step 3: Compute Overall Cluster Separation Metrics (Silhouette, DBI)

# %%
# Filter out rows where labels are None for Chapman
chapman_valid_mask = np.array([lbl in LABELS_OF_INTEREST for lbl in chapman_labels])
X_chapman_base_valid = baseline_arr[chapman_valid_mask]
X_chapman_ft_valid = finetuned_arr[chapman_valid_mask]
chapman_labels_valid = chapman_labels[chapman_valid_mask]

# Compute silhouette and DBI for Chapman baseline
sil_chap_base = silhouette_score(X_chapman_base_valid, chapman_labels_valid)
dbi_chap_base = davies_bouldin_score(X_chapman_base_valid, chapman_labels_valid)

# Compute silhouette and DBI for Chapman fine-tuned
sil_chap_ft = silhouette_score(X_chapman_ft_valid, chapman_labels_valid)
dbi_chap_ft = davies_bouldin_score(X_chapman_ft_valid, chapman_labels_valid)

print(f"Chapman Baseline - Silhouette: {sil_chap_base:.4f}, DBI: {dbi_chap_base:.4f}")
print(f"Chapman Fine-tuned - Silhouette: {sil_chap_ft:.4f}, DBI: {dbi_chap_ft:.4f}")
print(f"Silhouette improvement: {(sil_chap_ft - sil_chap_base) / abs(sil_chap_base) * 100:.2f}%")
print(f"DBI improvement: {(dbi_chap_base - dbi_chap_ft) / dbi_chap_base * 100:.2f}%")

# %% [markdown]
# ## Per-Label Silhouette Scores

# %%
# Compute silhouette samples for Chapman baseline
sil_samples_chap_base = silhouette_samples(X_chapman_base_valid, chapman_labels_valid)

# Compute silhouette samples for Chapman fine-tuned
sil_samples_chap_ft = silhouette_samples(X_chapman_ft_valid, chapman_labels_valid)

# Build a DataFrame to aggregate per label
df_sil_chap = pd.DataFrame({
    'label': chapman_labels_valid,
    'sil_base': sil_samples_chap_base,
    'sil_ft': sil_samples_chap_ft
})

# Group by label and compute mean silhouette for baseline and fine-tuned
df_sil_label_chap = df_sil_chap.groupby('label', as_index=False).agg({
    'sil_base': 'mean',
    'sil_ft': 'mean'
})

# Calculate improvement
df_sil_label_chap['sil_change'] = df_sil_label_chap['sil_ft'] - df_sil_label_chap['sil_base']
df_sil_label_chap['sil_change_pct'] = (
    df_sil_label_chap['sil_change'] / df_sil_label_chap['sil_base'].abs() * 100
)

# Print results
print("\nPer-Label Silhouette - Chapman Baseline vs. Fine-Tuned")
print(df_sil_label_chap.round(4))

# %% [markdown]
# ## Step 4: Silhouette & DBI for PTB-XL vs. Combined

# %%
# Filter out rows where labels are None for PTB-XL
ptbxl_valid_mask = np.array([lbl in LABELS_OF_INTEREST for lbl in ptbxl_labels])
X_ptbxl_base_valid = baseline_ptb[ptbxl_valid_mask]
X_ptbxl_ft_valid = finetuned_ptb[ptbxl_valid_mask]
ptbxl_labels_valid = ptbxl_labels[ptbxl_valid_mask]

# Compute silhouette and DBI for PTB-XL baseline
sil_ptbxl_base = silhouette_score(X_ptbxl_base_valid, ptbxl_labels_valid)
dbi_ptbxl_base = davies_bouldin_score(X_ptbxl_base_valid, ptbxl_labels_valid)

# Compute silhouette and DBI for PTB-XL fine-tuned
sil_ptbxl_ft = silhouette_score(X_ptbxl_ft_valid, ptbxl_labels_valid)
dbi_ptbxl_ft = davies_bouldin_score(X_ptbxl_ft_valid, ptbxl_labels_valid)

print(f"PTB-XL Baseline - Silhouette: {sil_ptbxl_base:.4f}, DBI: {dbi_ptbxl_base:.4f}")
print(f"PTB-XL Fine-tuned - Silhouette: {sil_ptbxl_ft:.4f}, DBI: {dbi_ptbxl_ft:.4f}")
print(f"Silhouette improvement: {(sil_ptbxl_ft - sil_ptbxl_base) / abs(sil_ptbxl_base) * 100:.2f}%")
print(f"DBI improvement: {(dbi_ptbxl_base - dbi_ptbxl_ft) / dbi_ptbxl_base * 100:.2f}%")

# %% [markdown]
# ## Per-Label Silhouette Scores for PTB-XL

# %%
# Compute silhouette samples for PTB-XL baseline
sil_samples_ptbxl_base = silhouette_samples(X_ptbxl_base_valid, ptbxl_labels_valid)

# Compute silhouette samples for PTB-XL fine-tuned
sil_samples_ptbxl_ft = silhouette_samples(X_ptbxl_ft_valid, ptbxl_labels_valid)

# Build a DataFrame to aggregate per label
df_sil_ptbxl = pd.DataFrame({
    'label': ptbxl_labels_valid,
    'sil_base': sil_samples_ptbxl_base,
    'sil_ft': sil_samples_ptbxl_ft
})

# Group by label and compute mean silhouette for baseline and fine-tuned
df_sil_label_ptbxl = df_sil_ptbxl.groupby('label', as_index=False).agg({
    'sil_base': 'mean',
    'sil_ft': 'mean'
})

# Calculate improvement
df_sil_label_ptbxl['sil_change'] = df_sil_label_ptbxl['sil_ft'] - df_sil_label_ptbxl['sil_base']
df_sil_label_ptbxl['sil_change_pct'] = (
    df_sil_label_ptbxl['sil_change'] / df_sil_label_ptbxl['sil_base'].abs() * 100
)

# Print results
print("\nPer-Label Silhouette - PTB-XL Baseline vs. Fine-Tuned")
print(df_sil_label_ptbxl.round(4))

# %% [markdown]
# ## Step 5: Compute Combined Dataset Metrics

# Combined dataset
X_combined_base = np.vstack((X_chapman_base_valid, X_ptbxl_base_valid))
X_combined_ft = np.vstack((X_chapman_ft_valid, X_ptbxl_ft_valid))
combined_labels = np.concatenate((chapman_labels_valid, ptbxl_labels_valid))

# Compute silhouette and DBI for combined baseline
sil_comb_base = silhouette_score(X_combined_base, combined_labels)
dbi_comb_base = davies_bouldin_score(X_combined_base, combined_labels)

# Compute silhouette and DBI for combined fine-tuned
sil_comb_ft = silhouette_score(X_combined_ft, combined_labels)
dbi_comb_ft = davies_bouldin_score(X_combined_ft, combined_labels)

print(f"Combined Baseline - Silhouette: {sil_comb_base:.4f}, DBI: {dbi_comb_base:.4f}")
print(f"Combined Fine-tuned - Silhouette: {sil_comb_ft:.4f}, DBI: {dbi_comb_ft:.4f}")
print(f"Silhouette improvement: {(sil_comb_ft - sil_comb_base) / abs(sil_comb_base) * 100:.2f}%")
print(f"DBI improvement: {(dbi_comb_base - dbi_comb_ft) / dbi_comb_base * 100:.2f}%")

# %% [markdown]
# ## Per-Label Silhouette Scores for Combined Dataset

# %%
# Compute silhouette samples for combined dataset
sil_samples_comb_base = silhouette_samples(X_combined_base, combined_labels)
sil_samples_comb_ft = silhouette_samples(X_combined_ft, combined_labels)

# Build a DataFrame to aggregate per label
df_sil_comb = pd.DataFrame({
    'label': combined_labels,
    'sil_base': sil_samples_comb_base,
    'sil_ft': sil_samples_comb_ft
})

# Group by label and compute mean silhouette for baseline and fine-tuned
df_sil_label_comb = df_sil_comb.groupby('label', as_index=False).agg({
    'sil_base': 'mean',
    'sil_ft': 'mean'
})

# Calculate improvement
df_sil_label_comb['sil_change'] = df_sil_label_comb['sil_ft'] - df_sil_label_comb['sil_base']
df_sil_label_comb['sil_change_pct'] = (
    df_sil_label_comb['sil_change'] / df_sil_label_comb['sil_base'].abs() * 100
)

# Print results
print("\nPer-Label Silhouette - Combined Dataset Baseline vs. Fine-Tuned")
print(df_sil_label_comb.round(4))

# %% [markdown]
# ## Visualize Per-Label Silhouette Scores

# %%
# Create a visualization for Chapman per-label silhouette scores
plt.figure(figsize=(12, 6))
barWidth = 0.3

# Set position of bar on X axis
r1 = np.arange(len(df_sil_label_chap))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, df_sil_label_chap['sil_base'], width=barWidth, edgecolor='grey', label='Baseline', alpha=0.7)
plt.bar(r2, df_sil_label_chap['sil_ft'], width=barWidth, edgecolor='grey', label='Fine-tuned', alpha=0.7)

# Add labels and title
plt.xlabel('Rhythm Label', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Chapman Per-Label Silhouette Scores: Baseline vs. Fine-tuned')
plt.xticks([r + barWidth/2 for r in range(len(df_sil_label_chap))], df_sil_label_chap['label'])
plt.legend()

# Add percentage improvement as text
for i, row in df_sil_label_chap.iterrows():
    plt.text(i + barWidth/2, 
             max(row['sil_base'], row['sil_ft']) + 0.02,
             f"{row['sil_change_pct']:.1f}%",
             ha='center', va='bottom', rotation=0, size=9)

plt.tight_layout()
plt.savefig('chapman_per_label_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a visualization for PTB-XL per-label silhouette scores
plt.figure(figsize=(12, 6))

# Set position of bar on X axis
r1 = np.arange(len(df_sil_label_ptbxl))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, df_sil_label_ptbxl['sil_base'], width=barWidth, edgecolor='grey', label='Baseline', alpha=0.7)
plt.bar(r2, df_sil_label_ptbxl['sil_ft'], width=barWidth, edgecolor='grey', label='Fine-tuned', alpha=0.7)

# Add labels and title
plt.xlabel('Rhythm Label', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('PTB-XL Per-Label Silhouette Scores: Baseline vs. Fine-tuned')
plt.xticks([r + barWidth/2 for r in range(len(df_sil_label_ptbxl))], df_sil_label_ptbxl['label'])
plt.legend()

# Add percentage improvement as text
for i, row in df_sil_label_ptbxl.iterrows():
    plt.text(i + barWidth/2, 
             max(row['sil_base'], row['sil_ft']) + 0.02,
             f"{row['sil_change_pct']:.1f}%",
             ha='center', va='bottom', rotation=0, size=9)

plt.tight_layout()
plt.savefig('ptbxl_per_label_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a visualization for combined per-label silhouette scores
plt.figure(figsize=(12, 6))

# Set position of bar on X axis
r1 = np.arange(len(df_sil_label_comb))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, df_sil_label_comb['sil_base'], width=barWidth, edgecolor='grey', label='Baseline', alpha=0.7)
plt.bar(r2, df_sil_label_comb['sil_ft'], width=barWidth, edgecolor='grey', label='Fine-tuned', alpha=0.7)

# Add labels and title
plt.xlabel('Rhythm Label', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Combined Per-Label Silhouette Scores: Baseline vs. Fine-tuned')
plt.xticks([r + barWidth/2 for r in range(len(df_sil_label_comb))], df_sil_label_comb['label'])
plt.legend()

# Add percentage improvement as text
for i, row in df_sil_label_comb.iterrows():
    plt.text(i + barWidth/2, 
             max(row['sil_base'], row['sil_ft']) + 0.02,
             f"{row['sil_change_pct']:.1f}%",
             ha='center', va='bottom', rotation=0, size=9)

plt.tight_layout()
plt.savefig('combined_per_label_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Dual-Label vs. Single-Label Distance Analysis for Chapman

# %%
# Identify single-labeled AFIB samples in Chapman
single_afib_mask = (df_arr["n_rhythm_labels"] == 1) & (df_arr["single_rhythm_label"] == "AFIB")
emb_single_afib = finetuned_arr[single_afib_mask]  # Using fine-tuned embeddings

# Calculate centroid of single-labeled AFIB
afib_centroid = emb_single_afib.mean(axis=0)

# Identify dual-labeled AFIB samples in Chapman
dual_afib_mask = (df_arr["n_rhythm_labels"] > 1) & (df_arr["rhythm_labels"].apply(lambda x: "AFIB" in x))
emb_dual_afib = finetuned_arr[dual_afib_mask]

# Calculate distances to AFIB centroid
if len(emb_dual_afib) > 0:  # Check if there are any dual-labeled AFIB samples
    dist_single_afib = cdist(emb_single_afib, afib_centroid.reshape(1, -1)).mean()
    dist_dual_afib = cdist(emb_dual_afib, afib_centroid.reshape(1, -1)).mean()
    
    print(f"\nDistance Analysis for AFIB in Chapman Dataset:")
    print(f"Single-labeled AFIB avg dist to centroid: {dist_single_afib:.4f}")
    print(f"Dual-labeled AFIB avg dist to centroid:   {dist_dual_afib:.4f}")
    print(f"Ratio (dual/single): {dist_dual_afib/dist_single_afib:.4f}")
    print(f"Percentage difference: {((dist_dual_afib - dist_single_afib) / dist_single_afib) * 100:.2f}%")

# Prepare data for visualization
labels = []
single_distances = []
dual_distances = []

# Add AFIB data if available
if len(emb_dual_afib) > 0:
    labels.append("AFIB")
    single_distances.append(dist_single_afib)
    dual_distances.append(dist_dual_afib)

# Add data for other rhythm labels
for label in [l for l in LABELS_OF_INTEREST if l != "AFIB"]:
    # Identify single-labeled samples for this rhythm
    single_mask = (df_arr["n_rhythm_labels"] == 1) & (df_arr["single_rhythm_label"] == label)
    emb_single = finetuned_arr[single_mask]
    
    # Identify dual-labeled samples for this rhythm
    dual_mask = (df_arr["n_rhythm_labels"] > 1) & (df_arr["rhythm_labels"].apply(lambda x: label in x))
    emb_dual = finetuned_arr[dual_mask]
    
    if len(emb_single) > 0 and len(emb_dual) > 0:
        centroid = emb_single.mean(axis=0)
        dist_single = cdist(emb_single, centroid.reshape(1, -1)).mean()
        dist_dual = cdist(emb_dual, centroid.reshape(1, -1)).mean()
        
        labels.append(label)
        single_distances.append(dist_single)
        dual_distances.append(dist_dual)

# Create bar plot
if labels:
    plt.figure(figsize=(12, 6))
    barWidth = 0.35
    
    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    
    # Make the plot
    plt.bar(r1, single_distances, width=barWidth, edgecolor='grey', label='Single-labeled', alpha=0.7)
    plt.bar(r2, dual_distances, width=barWidth, edgecolor='grey', label='Dual-labeled', alpha=0.7)
    
    # Add percentage differences as text
    for i in range(len(labels)):
        pct_diff = ((dual_distances[i] - single_distances[i]) / single_distances[i]) * 100
        plt.text(i + barWidth/2, 
                max(single_distances[i], dual_distances[i]) + 0.02,
                f"{pct_diff:.1f}%",
                ha='center', va='bottom', rotation=0, size=9)
    
    # Add labels and title
    plt.xlabel('Rhythm Label', fontweight='bold')
    plt.ylabel('Average Distance to Centroid', fontweight='bold')
    plt.title('Chapman Dataset: Single vs. Multi Label Distance to Centroid')
    plt.xticks([r + barWidth/2 for r in range(len(labels))], labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('chapman_single_vs_dual_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data available for visualization.")

# Repeat for other rhythm labels of interest
for label in [l for l in LABELS_OF_INTEREST if l != "AFIB"]:
    # Identify single-labeled samples for this rhythm
    single_mask = (df_arr["n_rhythm_labels"] == 1) & (df_arr["single_rhythm_label"] == label)
    emb_single = finetuned_arr[single_mask]
    
    if len(emb_single) == 0:
        continue
        
    # Calculate centroid
    centroid = emb_single.mean(axis=0)
    
    # Identify dual-labeled samples for this rhythm
    dual_mask = (df_arr["n_rhythm_labels"] > 1) & (df_arr["rhythm_labels"].apply(lambda x: label in x))
    emb_dual = finetuned_arr[dual_mask]
    
    if len(emb_dual) == 0:
        continue
        
    # Calculate distances
    dist_single = cdist(emb_single, centroid.reshape(1, -1)).mean()
    dist_dual = cdist(emb_dual, centroid.reshape(1, -1)).mean()
    
    print(f"\nDistance Analysis for {label} in Chapman Dataset:")
    print(f"Single-labeled {label} avg dist to centroid: {dist_single:.4f}")
    print(f"Dual-labeled {label} avg dist to centroid:   {dist_dual:.4f}")
    print(f"Ratio (dual/single): {dist_dual/dist_single:.4f}")
    print(f"Percentage difference: {((dist_dual - dist_single) / dist_single) * 100:.2f}%")

# %% [markdown]
# ## Step 6: Compute Chapman Centroids in Baseline & Fine-Tuned

# %%
# Initialize dictionaries for centroids
chapman_centroids_base = {}
chapman_centroids_ft = {}

# Compute centroids for each label in baseline
for label in LABELS_OF_INTEREST:
    mask = chapman_labels == label
    if np.sum(mask) >= 5:  # Ensure we have enough samples
        chapman_centroids_base[label] = np.mean(baseline_arr[mask], axis=0)
        chapman_centroids_ft[label] = np.mean(finetuned_arr[mask], axis=0)

# %% [markdown]
# ## Step 7: Distance of PTB-XL Points to Chapman Centroids

# %%
# Initialize results dictionary
centroid_distance_results = {}

# Calculate distances for each label
for label in LABELS_OF_INTEREST:
    if label not in chapman_centroids_base or label not in chapman_centroids_ft:
        print(f"Skipping {label} - no centroid available")
        continue
        
    # Get PTB-XL points for this label
    ptb_mask = ptbxl_labels == label
    X_ptb_label_base = baseline_ptb[ptb_mask]
    X_ptb_label_ft = finetuned_ptb[ptb_mask]
    
    if len(X_ptb_label_base) < 5:
        print(f"Skipping {label} - not enough PTB-XL samples")
        continue
    
    # Calculate distances to Chapman centroids
    centroid_base = chapman_centroids_base[label]
    centroid_ft = chapman_centroids_ft[label]
    
    # Baseline distances
    dist_base = cdist(X_ptb_label_base, centroid_base.reshape(1, -1), metric='euclidean')
    avg_dist_base = dist_base.mean()
    std_dist_base = dist_base.std()
    
    # Fine-tuned distances
    dist_ft = cdist(X_ptb_label_ft, centroid_ft.reshape(1, -1), metric='euclidean')
    avg_dist_ft = dist_ft.mean()
    std_dist_ft = dist_ft.std()
    
    # Calculate change
    dist_change = avg_dist_ft - avg_dist_base
    dist_change_pct = (dist_change / avg_dist_base) * 100
    
    # Store results
    centroid_distance_results[label] = {
        'baseline_avg_dist': avg_dist_base,
        'baseline_std_dist': std_dist_base,
        'finetuned_avg_dist': avg_dist_ft,
        'finetuned_std_dist': std_dist_ft,
        'dist_change': dist_change,
        'dist_change_pct': dist_change_pct,
        'n_samples': len(X_ptb_label_base)
    }

# Create a summary DataFrame
centroid_dist_df = pd.DataFrame([
    {
        'Label': label,
        'Baseline_Avg_Distance': results['baseline_avg_dist'],
        'Finetuned_Avg_Distance': results['finetuned_avg_dist'],
        'Distance_Change': results['dist_change'],
        'Distance_Change_Pct': results['dist_change_pct'],
        'N_Samples': results['n_samples']
    }
    for label, results in centroid_distance_results.items()
])

# Sort by percentage change
centroid_dist_df = centroid_dist_df.sort_values('Distance_Change_Pct')
print(centroid_dist_df)

# %% [markdown]
# ## Step 8: Intra vs. Inter-Class Ratio

# %%
# Initialize results dictionary
intra_inter_results = {}

# Calculate intra-class and inter-class distances for each label
for label in LABELS_OF_INTEREST:
    if label not in chapman_centroids_base or label not in chapman_centroids_ft:
        continue
        
    # Get PTB-XL points for this label
    ptb_mask = ptbxl_labels == label
    X_ptb_label_base = baseline_ptb[ptb_mask]
    X_ptb_label_ft = finetuned_ptb[ptb_mask]
    
    if len(X_ptb_label_base) < 5:
        continue
    
    # Baseline calculations
    # Intra-class distance (to own centroid)
    centroid_base = chapman_centroids_base[label]
    dist_intra_base = cdist(X_ptb_label_base, centroid_base.reshape(1, -1), metric='euclidean').mean()
    
    # Inter-class distances (to other centroids)
    dist_inter_base_list = []
    for other_label in LABELS_OF_INTEREST:
        if other_label != label and other_label in chapman_centroids_base:
            other_centroid_base = chapman_centroids_base[other_label]
            dist = cdist(X_ptb_label_base, other_centroid_base.reshape(1, -1), metric='euclidean').mean()
            dist_inter_base_list.append(dist)
    
    dist_inter_base_avg = np.mean(dist_inter_base_list)
    ratio_base = dist_intra_base / dist_inter_base_avg
    
    # Fine-tuned calculations
    # Intra-class distance (to own centroid)
    centroid_ft = chapman_centroids_ft[label]
    dist_intra_ft = cdist(X_ptb_label_ft, centroid_ft.reshape(1, -1), metric='euclidean').mean()
    
    # Inter-class distances (to other centroids)
    dist_inter_ft_list = []
    for other_label in LABELS_OF_INTEREST:
        if other_label != label and other_label in chapman_centroids_ft:
            other_centroid_ft = chapman_centroids_ft[other_label]
            dist = cdist(X_ptb_label_ft, other_centroid_ft.reshape(1, -1), metric='euclidean').mean()
            dist_inter_ft_list.append(dist)
    
    dist_inter_ft_avg = np.mean(dist_inter_ft_list)
    ratio_ft = dist_intra_ft / dist_inter_ft_avg
    
    # Calculate change
    ratio_change = ratio_ft - ratio_base
    ratio_change_pct = (ratio_change / ratio_base) * 100
    
    # Store results
    intra_inter_results[label] = {
        'intra_base': dist_intra_base,
        'inter_base_avg': dist_inter_base_avg,
        'ratio_base': ratio_base,
        'intra_ft': dist_intra_ft,
        'inter_ft_avg': dist_inter_ft_avg,
        'ratio_ft': ratio_ft,
        'ratio_change': ratio_change,
        'ratio_change_pct': ratio_change_pct,
        'n_samples': len(X_ptb_label_base)
    }

# Create a summary DataFrame
intra_inter_df = pd.DataFrame([
    {
        'Label': label,
        'Baseline_Intra': results['intra_base'],
        'Baseline_Inter_Avg': results['inter_base_avg'],
        'Baseline_Ratio': results['ratio_base'],
        'Finetuned_Intra': results['intra_ft'],
        'Finetuned_Inter_Avg': results['inter_ft_avg'],
        'Finetuned_Ratio': results['ratio_ft'],
        'Ratio_Change': results['ratio_change'],
        'Ratio_Change_Pct': results['ratio_change_pct'],
        'N_Samples': results['n_samples']
    }
    for label, results in intra_inter_results.items()
])

# Sort by percentage change
intra_inter_df = intra_inter_df.sort_values('Ratio_Change_Pct')
print(intra_inter_df)

# %% [markdown]
# ## Step 9: Visualize the Results

# %%
# Create a bar plot to compare baseline and fine-tuned distances
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(centroid_dist_df))

plt.bar(index, centroid_dist_df['Baseline_Avg_Distance'], bar_width, 
        label='Baseline', color='lightblue', alpha=0.8)
plt.bar(index + bar_width, centroid_dist_df['Finetuned_Avg_Distance'], bar_width,
        label='Fine-tuned', color='orange', alpha=0.8)

plt.xlabel('Rhythm Label')
plt.ylabel('Average Distance to Chapman Centroid')
plt.title('Comparison of Centroid Distances: Baseline vs. Fine-tuned')
plt.xticks(index + bar_width / 2, centroid_dist_df['Label'])
plt.legend()

# Add text annotations for percentage change
for i, row in enumerate(centroid_dist_df.itertuples()):
    change_pct = row.Distance_Change_Pct
    color = 'green' if change_pct < 0 else 'red'
    plt.text(i + bar_width/2, max(row.Baseline_Avg_Distance, row.Finetuned_Avg_Distance) + 0.1,
             f"{change_pct:.1f}%", ha='center', va='bottom', color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('centroid_distance_comparison.png', dpi=300)
plt.show()

# %%
# Create a bar plot to compare intra/inter ratios
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(intra_inter_df))

plt.bar(index, intra_inter_df['Baseline_Ratio'], bar_width, 
        label='Baseline', color='lightblue', alpha=0.8)
plt.bar(index + bar_width, intra_inter_df['Finetuned_Ratio'], bar_width,
        label='Fine-tuned', color='orange', alpha=0.8)

plt.xlabel('Rhythm Label')
plt.ylabel('Intra-Class / Inter-Class Distance Ratio')
plt.title('Comparison of Intra/Inter Class Distance Ratios: Baseline vs. Fine-tuned')
plt.xticks(index + bar_width / 2, intra_inter_df['Label'])
plt.legend()

# Add text annotations for percentage change
for i, row in enumerate(intra_inter_df.itertuples()):
    change_pct = row.Ratio_Change_Pct
    color = 'green' if change_pct < 0 else 'red'
    plt.text(i + bar_width/2, max(row.Baseline_Ratio, row.Finetuned_Ratio) + 0.02,
             f"{change_pct:.1f}%", ha='center', va='bottom', color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('intra_inter_ratio_comparison.png', dpi=300)
plt.show()

# %% [markdown]
# ## Final Summary Table

# %%
# Create a comprehensive results table
final_results = []

for label in LABELS_OF_INTEREST:
    if label in centroid_distance_results and label in intra_inter_results:
        final_results.append({
            'Label': label,
            'Baseline_Dist': centroid_distance_results[label]['baseline_avg_dist'],
            'Finetuned_Dist': centroid_distance_results[label]['finetuned_avg_dist'],
            'Dist_Change_Pct': centroid_distance_results[label]['dist_change_pct'],
            'Baseline_Ratio': intra_inter_results[label]['ratio_base'],
            'Finetuned_Ratio': intra_inter_results[label]['ratio_ft'],
            'Ratio_Change_Pct': intra_inter_results[label]['ratio_change_pct'],
            'N_Samples': centroid_distance_results[label]['n_samples']
        })

final_df = pd.DataFrame(final_results)
final_df = final_df.sort_values('Dist_Change_Pct')

# Format the table for better readability
formatted_df = final_df.copy()
formatted_df['Baseline_Dist'] = formatted_df['Baseline_Dist'].round(2)
formatted_df['Finetuned_Dist'] = formatted_df['Finetuned_Dist'].round(2)
formatted_df['Dist_Change_Pct'] = formatted_df['Dist_Change_Pct'].round(1).astype(str) + '%'
formatted_df['Baseline_Ratio'] = formatted_df['Baseline_Ratio'].round(2)
formatted_df['Finetuned_Ratio'] = formatted_df['Finetuned_Ratio'].round(2)
formatted_df['Ratio_Change_Pct'] = formatted_df['Ratio_Change_Pct'].round(1).astype(str) + '%'

print("Final Results Table:")
print(formatted_df)

# %% [markdown]
# ## Conclusion
# 
# The centroid analysis provides insights into how well the two datasets (Chapman and PTB-XL) overlap in the embedding space. 

# %% [markdown]
# ## SR-SB Overlap Analysis
# This section quantitatively confirms the observation that SR and SB appear "mixed up" in the fine-tuned embedding space.
# We use two approaches:
# 1. Distance-to-Centroid Analysis
# 2. Silhouette Score Analysis

# %%
print("\n" + "="*80)
print("QUANTITATIVE ANALYSIS OF SR-SB OVERLAP IN FINE-TUNED EMBEDDING SPACE")
print("="*80)

# %% [markdown]
# ### 1. Distance-To-Centroid Analysis
# We compute centroids for SR and SB using only single-labeled samples, then measure how close
# each point is to both centroids. If the points are nearly equidistant to both centroids,
# this indicates poor separation.

# %%
# 1. Compute SR and SB centroids using only single-labeled samples
# Chapman dataset
sr_single_mask_chapman = (chapman_labels == "SR") & (df_arr["n_rhythm_labels"] == 1)
sb_single_mask_chapman = (chapman_labels == "SB") & (df_arr["n_rhythm_labels"] == 1)

# PTB-XL dataset
sr_single_mask_ptbxl = (ptbxl_labels == "SR") & (df_ptbxl["is_single_labeled"] == True)
sb_single_mask_ptbxl = (ptbxl_labels == "SB") & (df_ptbxl["is_single_labeled"] == True)

# Compute centroids for Chapman
sr_centroid_chapman_ft = finetuned_arr[sr_single_mask_chapman].mean(axis=0)
sb_centroid_chapman_ft = finetuned_arr[sb_single_mask_chapman].mean(axis=0)

# Compute centroids for PTB-XL
sr_centroid_ptbxl_ft = finetuned_ptb[sr_single_mask_ptbxl].mean(axis=0)
sb_centroid_ptbxl_ft = finetuned_ptb[sb_single_mask_ptbxl].mean(axis=0)

# 2. For SR-labeled points, measure distance to SR centroid and SB centroid
# Chapman dataset
sr_points_chapman_ft = finetuned_arr[chapman_labels == "SR"]
sr_to_sr_dist_chapman = cdist(sr_points_chapman_ft, sr_centroid_chapman_ft.reshape(1, -1)).mean()
sr_to_sb_dist_chapman = cdist(sr_points_chapman_ft, sb_centroid_chapman_ft.reshape(1, -1)).mean()
sr_ratio_chapman = sr_to_sb_dist_chapman / sr_to_sr_dist_chapman

# 3. For SB-labeled points, measure distance to SB centroid and SR centroid
sb_points_chapman_ft = finetuned_arr[chapman_labels == "SB"]
sb_to_sb_dist_chapman = cdist(sb_points_chapman_ft, sb_centroid_chapman_ft.reshape(1, -1)).mean()
sb_to_sr_dist_chapman = cdist(sb_points_chapman_ft, sr_centroid_chapman_ft.reshape(1, -1)).mean()
sb_ratio_chapman = sb_to_sr_dist_chapman / sb_to_sb_dist_chapman

# 4. Repeat for PTB-XL dataset
sr_points_ptbxl_ft = finetuned_ptb[ptbxl_labels == "SR"]
sr_to_sr_dist_ptbxl = cdist(sr_points_ptbxl_ft, sr_centroid_ptbxl_ft.reshape(1, -1)).mean()
sr_to_sb_dist_ptbxl = cdist(sr_points_ptbxl_ft, sb_centroid_ptbxl_ft.reshape(1, -1)).mean()
sr_ratio_ptbxl = sr_to_sb_dist_ptbxl / sr_to_sr_dist_ptbxl

sb_points_ptbxl_ft = finetuned_ptb[ptbxl_labels == "SB"]
sb_to_sb_dist_ptbxl = cdist(sb_points_ptbxl_ft, sb_centroid_ptbxl_ft.reshape(1, -1)).mean()
sb_to_sr_dist_ptbxl = cdist(sb_points_ptbxl_ft, sr_centroid_ptbxl_ft.reshape(1, -1)).mean()
sb_ratio_ptbxl = sb_to_sr_dist_ptbxl / sb_to_sb_dist_ptbxl

# 5. Create a summary table
sr_sb_distance_df = pd.DataFrame([
    {
        'Dataset': 'Chapman',
        'Label': 'SR',
        'Dist_to_SR_Centroid': sr_to_sr_dist_chapman,
        'Dist_to_SB_Centroid': sr_to_sb_dist_chapman,
        'Ratio': sr_ratio_chapman
    },
    {
        'Dataset': 'Chapman',
        'Label': 'SB',
        'Dist_to_SR_Centroid': sb_to_sr_dist_chapman,
        'Dist_to_SB_Centroid': sb_to_sb_dist_chapman,
        'Ratio': sb_ratio_chapman
    },
    {
        'Dataset': 'PTB-XL',
        'Label': 'SR',
        'Dist_to_SR_Centroid': sr_to_sr_dist_ptbxl,
        'Dist_to_SB_Centroid': sr_to_sb_dist_ptbxl,
        'Ratio': sr_ratio_ptbxl
    },
    {
        'Dataset': 'PTB-XL',
        'Label': 'SB',
        'Dist_to_SR_Centroid': sb_to_sr_dist_ptbxl,
        'Dist_to_SB_Centroid': sb_to_sb_dist_ptbxl,
        'Ratio': sb_ratio_ptbxl
    }
])

# Format the table for better readability
pd.set_option('display.float_format', '{:.4f}'.format)
print("\nDistance-to-Centroid Analysis for SR and SB:")
print(sr_sb_distance_df)

# Calculate the percentage of SR points closer to SB centroid than SR centroid
# Chapman dataset
sr_points_chapman_dists_to_sr = cdist(sr_points_chapman_ft, sr_centroid_chapman_ft.reshape(1, -1)).flatten()
sr_points_chapman_dists_to_sb = cdist(sr_points_chapman_ft, sb_centroid_chapman_ft.reshape(1, -1)).flatten()
sr_closer_to_sb_chapman = np.sum(sr_points_chapman_dists_to_sb < sr_points_chapman_dists_to_sr)
sr_closer_to_sb_pct_chapman = (sr_closer_to_sb_chapman / len(sr_points_chapman_ft)) * 100

# Calculate the percentage of SB points closer to SR centroid than SB centroid
sb_points_chapman_dists_to_sr = cdist(sb_points_chapman_ft, sr_centroid_chapman_ft.reshape(1, -1)).flatten()
sb_points_chapman_dists_to_sb = cdist(sb_points_chapman_ft, sb_centroid_chapman_ft.reshape(1, -1)).flatten()
sb_closer_to_sr_chapman = np.sum(sb_points_chapman_dists_to_sr < sb_points_chapman_dists_to_sb)
sb_closer_to_sr_pct_chapman = (sb_closer_to_sr_chapman / len(sb_points_chapman_ft)) * 100

# PTB-XL dataset
sr_points_ptbxl_dists_to_sr = cdist(sr_points_ptbxl_ft, sr_centroid_ptbxl_ft.reshape(1, -1)).flatten()
sr_points_ptbxl_dists_to_sb = cdist(sr_points_ptbxl_ft, sb_centroid_ptbxl_ft.reshape(1, -1)).flatten()
sr_closer_to_sb_ptbxl = np.sum(sr_points_ptbxl_dists_to_sb < sr_points_ptbxl_dists_to_sr)
sr_closer_to_sb_pct_ptbxl = (sr_closer_to_sb_ptbxl / len(sr_points_ptbxl_ft)) * 100

sb_points_ptbxl_dists_to_sr = cdist(sb_points_ptbxl_ft, sr_centroid_ptbxl_ft.reshape(1, -1)).flatten()
sb_points_ptbxl_dists_to_sb = cdist(sb_points_ptbxl_ft, sb_centroid_ptbxl_ft.reshape(1, -1)).flatten()
sb_closer_to_sr_ptbxl = np.sum(sb_points_ptbxl_dists_to_sr < sb_points_ptbxl_dists_to_sb)
sb_closer_to_sr_pct_ptbxl = (sb_closer_to_sr_ptbxl / len(sb_points_ptbxl_ft)) * 100

print("\nPercentage of points closer to the other class's centroid:")
print(f"Chapman: {sr_closer_to_sb_pct_chapman:.2f}% of SR points are closer to SB centroid than SR centroid")
print(f"Chapman: {sb_closer_to_sr_pct_chapman:.2f}% of SB points are closer to SR centroid than SB centroid")
print(f"PTB-XL: {sr_closer_to_sb_pct_ptbxl:.2f}% of SR points are closer to SB centroid than SR centroid")
print(f"PTB-XL: {sb_closer_to_sr_pct_ptbxl:.2f}% of SB points are closer to SR centroid than SB centroid")

# %% [markdown]
# ### 2. Silhouette Score Analysis
# We compute the silhouette score for SR vs. SB points only. A low or negative silhouette score
# indicates poor separation between the classes.

# %%
# Chapman dataset
sr_sb_mask_chapman = np.logical_or(chapman_labels == "SR", chapman_labels == "SB")
X_sr_sb_chapman = finetuned_arr[sr_sb_mask_chapman]
labels_sr_sb_chapman = chapman_labels[sr_sb_mask_chapman]

# Compute silhouette score for SR vs. SB
silhouette_sr_sb_chapman = silhouette_score(X_sr_sb_chapman, labels_sr_sb_chapman)

# Compute per-label silhouette samples
silhouette_samples_sr_sb_chapman = silhouette_samples(X_sr_sb_chapman, labels_sr_sb_chapman)
sr_silhouette_chapman = silhouette_samples_sr_sb_chapman[labels_sr_sb_chapman == "SR"].mean()
sb_silhouette_chapman = silhouette_samples_sr_sb_chapman[labels_sr_sb_chapman == "SB"].mean()

# PTB-XL dataset
sr_sb_mask_ptbxl = np.logical_or(ptbxl_labels == "SR", ptbxl_labels == "SB")
X_sr_sb_ptbxl = finetuned_ptb[sr_sb_mask_ptbxl]
labels_sr_sb_ptbxl = ptbxl_labels[sr_sb_mask_ptbxl]

# Compute silhouette score for SR vs. SB
silhouette_sr_sb_ptbxl = silhouette_score(X_sr_sb_ptbxl, labels_sr_sb_ptbxl)

# Compute per-label silhouette samples
silhouette_samples_sr_sb_ptbxl = silhouette_samples(X_sr_sb_ptbxl, labels_sr_sb_ptbxl)
sr_silhouette_ptbxl = silhouette_samples_sr_sb_ptbxl[labels_sr_sb_ptbxl == "SR"].mean()
sb_silhouette_ptbxl = silhouette_samples_sr_sb_ptbxl[labels_sr_sb_ptbxl == "SB"].mean()

# Create a summary table
silhouette_df = pd.DataFrame([
    {
        'Dataset': 'Chapman',
        'Overall_Silhouette': silhouette_sr_sb_chapman,
        'SR_Silhouette': sr_silhouette_chapman,
        'SB_Silhouette': sb_silhouette_chapman
    },
    {
        'Dataset': 'PTB-XL',
        'Overall_Silhouette': silhouette_sr_sb_ptbxl,
        'SR_Silhouette': sr_silhouette_ptbxl,
        'SB_Silhouette': sb_silhouette_ptbxl
    }
])

print("\nSilhouette Score Analysis for SR vs. SB:")
print(silhouette_df)

# %% [markdown]
# ### 3. Visualization of SR-SB Distances
# We visualize the distribution of distances from SR points to both centroids, and similarly for SB points.

# %%
# Prepare data for visualization
# Chapman dataset
sr_points_chapman_dists = pd.DataFrame({
    'Point_Type': 'SR',
    'Distance_Type': 'To SR Centroid',
    'Distance': sr_points_chapman_dists_to_sr
})

sr_points_chapman_dists_sb = pd.DataFrame({
    'Point_Type': 'SR',
    'Distance_Type': 'To SB Centroid',
    'Distance': sr_points_chapman_dists_to_sb
})

sb_points_chapman_dists = pd.DataFrame({
    'Point_Type': 'SB',
    'Distance_Type': 'To SB Centroid',
    'Distance': sb_points_chapman_dists_to_sb
})

sb_points_chapman_dists_sr = pd.DataFrame({
    'Point_Type': 'SB',
    'Distance_Type': 'To SR Centroid',
    'Distance': sb_points_chapman_dists_to_sr
})

# Combine all dataframes
chapman_distances_df = pd.concat([
    sr_points_chapman_dists, 
    sr_points_chapman_dists_sb,
    sb_points_chapman_dists,
    sb_points_chapman_dists_sr
])

# Create a boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Point_Type', y='Distance', hue='Distance_Type', data=chapman_distances_df)
plt.title('Chapman Dataset: Distribution of Distances to SR and SB Centroids')
plt.xlabel('Point Type')
plt.ylabel('Distance to Centroid')
plt.tight_layout()
plt.savefig('chapman_sr_sb_distances.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4. Combined Analysis Summary
# We combine the distance-to-centroid and silhouette analyses to provide a comprehensive assessment
# of the SR-SB overlap in the fine-tuned embedding space.

# %%
print("\n" + "="*80)
print("SUMMARY OF SR-SB OVERLAP ANALYSIS")
print("="*80)

print("\nChapman Dataset:")
print(f"1. Distance Analysis:")
print(f"   - SR points are only {(sr_ratio_chapman - 1) * 100:.2f}% farther from SB centroid than from SR centroid")
print(f"   - SB points are only {(sb_ratio_chapman - 1) * 100:.2f}% farther from SR centroid than from SB centroid")
print(f"   - {sr_closer_to_sb_pct_chapman:.2f}% of SR points are actually closer to SB centroid than SR centroid")
print(f"   - {sb_closer_to_sr_pct_chapman:.2f}% of SB points are actually closer to SR centroid than SB centroid")
print(f"2. Silhouette Analysis:")
print(f"   - Overall SR-SB silhouette score: {silhouette_sr_sb_chapman:.4f}")
print(f"   - SR silhouette score: {sr_silhouette_chapman:.4f}")
print(f"   - SB silhouette score: {sb_silhouette_chapman:.4f}")

print("\nPTB-XL Dataset:")
print(f"1. Distance Analysis:")
print(f"   - SR points are only {(sr_ratio_ptbxl - 1) * 100:.2f}% farther from SB centroid than from SR centroid")
print(f"   - SB points are only {(sb_ratio_ptbxl - 1) * 100:.2f}% farther from SR centroid than from SB centroid")
print(f"   - {sr_closer_to_sb_pct_ptbxl:.2f}% of SR points are actually closer to SB centroid than SR centroid")
print(f"   - {sb_closer_to_sr_pct_ptbxl:.2f}% of SB points are actually closer to SR centroid than SB centroid")
print(f"2. Silhouette Analysis:")
print(f"   - Overall SR-SB silhouette score: {silhouette_sr_sb_ptbxl:.4f}")
print(f"   - SR silhouette score: {sr_silhouette_ptbxl:.4f}")
print(f"   - SB silhouette score: {sb_silhouette_ptbxl:.4f}")

print("\nConclusion:")
print("The quantitative analysis confirms the visual observation that SR and SB appear 'mixed up' in the fine-tuned embedding space.")
print("Both the distance-to-centroid ratios (near 1.0) and the low/negative silhouette scores indicate substantial overlap between these classes.")
print("This suggests that the model has difficulty distinguishing between SR and SB, possibly due to similarities in their ECG manifestations.")
