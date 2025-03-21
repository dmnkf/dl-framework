# %% Phase 3: Cross-Domain Generalization (PTB-XL)
"""
Phase 3: Exploring PTB-XL Dataset Metadata
------------------------------------------
In this phase, we want to see if our fine-tuned encoder (trained on the
arrhythmia dataset) can generalize to another ECG dataset: PTB-XL.

Step 1: Investigate PTB-XL metadata structure:
 - Load PTB-XL from UnifiedDataset with "ptbxl" as dataset_key.
 - Explore how the metadata (labels, groups, etc.) are stored compared
   to the arrhythmia dataset.
 - Print a few samples to understand the label schemas, so we can later
   map them to arrhythmia labels for color alignment and contour plotting.
"""

import sys
from pathlib import Path

# We assume project_root and other configurations are already defined above.
# We'll just re-use them:
ptbxl_data = UnifiedDataset(
    Path(project_root) / "data", modality=DatasetModality.ECG, dataset_key="ptbxl"
)

# we can use all as no training was done on this dataset
ptbxl_ids = ptbxl_data.get_splits().get("test")
ptbxl_md_store = ptbxl_data.metadata_store


print("PTB-XL dataset loaded.")
print(f"Number of records: {len(ptbxl_ids)}")

sample_ids = ptbxl_ids[:5]  # first 5 for demonstration
print("\nInspecting metadata for sample records in PTB-XL set:")

for rid in sample_ids:
    meta = ptbxl_md_store.get(rid, {})
    print("-" * 70)
    print(f"Record ID: {rid}")
    # We'll print out the keys so we see how the structure is organized
    # For instance: 'labels_metadata' might contain label info
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print()

print("Done inspecting PTB-XL metadata structure.")

# %% Phase 3 (continued): Investigating PTB-XL SCP Codes & Statements
"""
We know from the Phase 3 introduction that PTB-XL label info is stored
in `scp_codes` (e.g., {'LVH': 50.0, 'SR': 0.0}) and in `scp_statements` 
(where each key, e.g., 'LVH', 'SR', has detailed meta-data).

Goal of this cell:
 - Collect & explore all unique scp_codes and scp_statements from PTB-XL
   test set to see how we can map them to our "groups" (Rhythm, Morphology,
   Amplitude, Duration, etc.) used in the arrhythmia dataset.

We'll produce a summary DataFrame of all scp codes encountered:
   + scp_code
   + # occurrences
   + fields like 'diagnostic_class', 'diagnostic_subclass', 'rhythm', etc.

Later, we'll define how to map them to our existing groups.
"""

import pandas as pd
from collections import defaultdict

ptbxl_scp_summary = []

for rid in ptbxl_ids:
    meta = ptbxl_md_store.get(rid, {})
    scp_statements = meta.get("scp_statements", {})

    # Each scp_code is e.g. 'LVH', 'SR', 'NORM', etc.
    for scp_code, scp_info in scp_statements.items():
        # We'll store a row with the record_id plus the scp_code's key properties
        row = {
            "record_id": rid,
            "scp_code": scp_code,
            # Some codes have these fields, some may be None.
            "description": scp_info.get("description", ""),
            "diagnostic": scp_info.get("diagnostic", None),
            "rhythm": scp_info.get("rhythm", None),
            "form": scp_info.get("form", None),
            "diagnostic_class": scp_info.get("diagnostic_class", None),
            "diagnostic_subclass": scp_info.get("diagnostic_subclass", None),
            "statement_category": scp_info.get("Statement Category", None),
        }
        ptbxl_scp_summary.append(row)

# Convert to DataFrame
df_ptbxl_scp = pd.DataFrame(ptbxl_scp_summary)

print(f"Unique scp_codes encountered: {df_ptbxl_scp['scp_code'].nunique()}")
print("Some sample rows:")
display(df_ptbxl_scp.head(10))

# Now let's see how frequently each scp_code occurs:
code_counts = df_ptbxl_scp["scp_code"].value_counts().reset_index()
code_counts.columns = ["scp_code", "count"]
print("\nTop 20 most common scp_codes in test set:")
display(code_counts.head(20))

# We might also want to see a pivot on diagnostic_class or rhythm=1
diagn_class_counts = df_ptbxl_scp["diagnostic_class"].value_counts(dropna=False)
print("\nDistribution of 'diagnostic_class':")
print(diagn_class_counts)

print("\nSample of scp_codes grouped by 'diagnostic_class':")
grouped_classes = df_ptbxl_scp.groupby("diagnostic_class")["scp_code"].unique()
for cls_name, codes in grouped_classes.items():
    print(f"- {cls_name}: {list(codes)}")

print("\nDone exploring PTB-XL scp_statements. Next step: create a mapping.")
# %% Phase 3 (continued): Checking SR vs. NORM presence in PTB-XL
"""
We observed a pattern where 'SR' (sinus rhythm) isn't necessarily
labeled as 'NORM' (normal ECG). Let's look for:
1) Records containing 'SR' but no 'NORM'
2) Records containing 'NORM' but no 'SR'
We then print the full metadata for those records to see how they differ.
"""

# 1) Group scp_codes by record_id into a list
df_codes_by_rec = (
    df_ptbxl_scp.groupby("record_id")["scp_code"]
    .agg(list)
    .reset_index(name="codes_list")
)

# 2) Add boolean flags
df_codes_by_rec["has_SR"] = df_codes_by_rec["codes_list"].apply(lambda x: "SR" in x)
df_codes_by_rec["has_NORM"] = df_codes_by_rec["codes_list"].apply(lambda x: "NORM" in x)

# 3) Find record_ids matching our criteria
records_with_sr_not_norm = df_codes_by_rec[
    (df_codes_by_rec["has_SR"]) & (~df_codes_by_rec["has_NORM"])
]["record_id"].tolist()

records_with_norm_not_sr = df_codes_by_rec[
    (df_codes_by_rec["has_NORM"]) & (~df_codes_by_rec["has_SR"])
]["record_id"].tolist()

print(f"# of records with SR but NOT NORM: {len(records_with_sr_not_norm)}")
print(f"# of records with NORM but NOT SR: {len(records_with_norm_not_sr)}")

# 4) Print sample metadata for each group
#    We'll just print the first 5 for demonstration; you can adjust as needed.


def print_full_metadata_for(rid_list, label_desc, max_print=5):
    print("\n" + "-" * 70)
    print(f"Printing {label_desc} (up to {max_print} records):")
    for rid in rid_list[:max_print]:
        meta = ptbxl_md_store.get(rid, {})
        print(f"\nRecord ID: {rid}")
        print(f"scp_codes: {meta.get('scp_codes', {})}")
        print(f"scp_statements: {meta.get('scp_statements', {})}")
        print("Full metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")


print_full_metadata_for(records_with_sr_not_norm, "Records with SR but NOT NORM")
print_full_metadata_for(records_with_norm_not_sr, "Records with NORM but NOT SR")

# %% Phase 3 (continued): Building a Multi-Label Structure for PTB-XL
"""
We want a consistent, per-record multi-label representation, akin to what
we did with arrhythmia. Each row in 'df_ptbxl_records' will represent 
a single record and list all "active" (prob == 100) labels. 

Later, we'll map these 'scp_codes' to the group structure used in arrhythmia
(e.g., 'Rhythm', 'Morphology', etc.). 
"""

ptbxl_records_info = []

for rid in ptbxl_ids:
    meta = ptbxl_md_store.get(rid, {})

    # scp_codes might be: {'IMI': 100.0, 'LVH': 50.0, 'SR': 0.0, ...}
    # Filter to codes with prob == 100
    scp_codes_dict = meta.get("scp_codes", {})
    active_codes = [code for code, prob in scp_codes_dict.items() if prob == 100.0]

    # For each active code, gather metadata from scp_statements
    scp_statements = meta.get("scp_statements", {})

    labels_meta = []
    for code in active_codes:
        info = scp_statements.get(code, {})
        # We'll store a dictionary of relevant fields.
        # Later, this can be mapped to arrhythmia-like groups.
        label_dict = {
            "scp_code": code,
            "description": info.get("description", ""),
            "diagnostic_class": info.get("diagnostic_class", None),
            "diagnostic_subclass": info.get("diagnostic_subclass", None),
            "rhythm": info.get("rhythm", None),
        }
        labels_meta.append(label_dict)

    ptbxl_records_info.append(
        {
            "record_id": rid,
            "active_codes": active_codes,  # list of codes with prob==100
            "labels_meta": labels_meta,  # structured metadata for each code
            "n_labels": len(active_codes),
        }
    )

df_ptbxl_records = pd.DataFrame(ptbxl_records_info)
df_ptbxl_records["row_idx"] = (
    df_ptbxl_records.index
)  # so we can track row <-> embedding

print("PTB-XL multi-label structure created.")
print(f"Total test records: {len(df_ptbxl_records)}")
display(df_ptbxl_records.head(10))

# Let's also see how many are single-labeled vs. multi-labeled:
n_single = (df_ptbxl_records["n_labels"] == 1).sum()
n_multi = (df_ptbxl_records["n_labels"] > 1).sum()

print(f"\nSingle-labeled records: {n_single}")
print(f"Multi-labeled records: {n_multi}")


# %% Phase 3 (continued): SCP Statement Breakdown (diagnostic, rhythm, form)
"""
PTB-XL scp_statements often have 3 potential flags:
 - diagnostic=1.0
 - rhythm=1.0
 - form=1.0

Any single scp_code can have one or more of these flags set (or none).
In this cell:
1) We'll gather all scp_statements from the test set (regardless of prob).
2) For each scp_code, create boolean flags: is_diagnostic, is_rhythm, is_form.
3) Summarize how often each flag combination appears across the entire test set.
4) Identify how many records have exactly one rhythm statement vs. multiple.

Note: Here, we do NOT filter on scp_codes with prob==100; we want
a full breakdown to understand the dataset. You can adapt if you
only want to consider "active" codes.
"""

import pandas as pd

records_scp_details = []  # Each row = (record_id, scp_code, is_diagnostic, is_rhythm, is_form)

for rid in ptbxl_ids:
    meta = ptbxl_md_store.get(rid, {})
    scp_codes = meta.get("scp_codes", {})  # e.g. {'IMI':100.0, 'LVH':50.0, ...}
    scp_stmts = meta.get("scp_statements", {})  # e.g. {'IMI':{...}, 'LVH':{...}, ...}

    for scp_code, stmt_info in scp_stmts.items():
        # Flags
        diag_val = stmt_info.get("diagnostic", 0.0)  # 1.0 or NaN
        rhythm_val = stmt_info.get("rhythm", 0.0)    # 1.0 or NaN
        form_val = stmt_info.get("form", 0.0)        # 1.0 or NaN

        is_diagnostic = (diag_val == 1.0)
        is_rhythm = (rhythm_val == 1.0)
        is_form = (form_val == 1.0)

        records_scp_details.append({
            "record_id": rid,
            "scp_code": scp_code,
            "is_diagnostic": is_diagnostic,
            "is_rhythm": is_rhythm,
            "is_form": is_form,
            # optional: also store scp_codes[scp_code] prob, if needed:
            "prob": scp_codes.get(scp_code, None),
        })

df_scp_breakdown = pd.DataFrame(records_scp_details)
print("Shape of df_scp_breakdown:", df_scp_breakdown.shape)
display(df_scp_breakdown.head(10))

# --------------------------------------------------------------------
# 1) Overall frequency of each flag
# --------------------------------------------------------------------
flag_counts = {
    "diagnostic=1": df_scp_breakdown["is_diagnostic"].sum(),
    "rhythm=1": df_scp_breakdown["is_rhythm"].sum(),
    "form=1": df_scp_breakdown["is_form"].sum(),
}
print("\nFrequency of statements with each flag set to True:")
for k, v in flag_counts.items():
    print(f"  {k}: {int(v)}")

# --------------------------------------------------------------------
# 2) Combination of flags per statement
#    For example, a code might have is_rhythm=True AND is_diagnostic=True
# --------------------------------------------------------------------
df_scp_breakdown["flag_combo"] = df_scp_breakdown.apply(
    lambda row: (
        f"{'D' if row.is_diagnostic else ''}"
        f"{'R' if row.is_rhythm else ''}"
        f"{'F' if row.is_form else ''}"
    ),
    axis=1
)

combo_counts = df_scp_breakdown["flag_combo"].value_counts()
print("\nHow many statements fall into each combination (D=diagnostic, R=rhythm, F=form)?")
display(combo_counts)

# --------------------------------------------------------------------
# 3) Records that contain EXACTLY one rhythm statement
#    We'll group by record_id to count how many scp_codes have is_rhythm=True.
# --------------------------------------------------------------------
df_rhythm_counts = (
    df_scp_breakdown
    .groupby("record_id")["is_rhythm"]
    .sum()
    .reset_index(name="n_rhythm_statements")
)

df_one_rhythm = df_rhythm_counts[df_rhythm_counts["n_rhythm_statements"] == 1]
df_multi_rhythm = df_rhythm_counts[df_rhythm_counts["n_rhythm_statements"] > 1]
df_no_rhythm = df_rhythm_counts[df_rhythm_counts["n_rhythm_statements"] == 0]

print(f"\nRecords with exactly ONE rhythm statement: {len(df_one_rhythm)}")
print(f"Records with MULTIPLE rhythm statements:   {len(df_multi_rhythm)}")
print(f"Records with NO rhythm statements:         {len(df_no_rhythm)}")

print("\nSample of record_ids with exactly one rhythm statement:")
display(df_one_rhythm.head(10))

# You could optionally merge these record_ids back to the full metadata
# or print them if you want a deeper inspection.
# %% Phase 3 (continued): Records with EXACTLY ONE rhythm statement, no other statements
"""
We want to find how many records have exactly 1 total statement (scp_code),
and that single statement is flagged as is_rhythm=True.

Steps:
1. Group by record_id to count the total number of statements (scp_codes).
2. Also sum the 'is_rhythm' flag to see how many rhythm statements each record has.
3. Filter where total statements == 1 AND total rhythm statements == 1.
"""

# 1) Group by record_id
df_statement_counts = (
    df_scp_breakdown
    .groupby("record_id")
    .agg(
        total_statements=("scp_code", "count"),
        n_rhythm_statements=("is_rhythm", "sum")
    )
    .reset_index()
)

# 2) Filter for records with exactly 1 total statement and that statement is a rhythm statement
df_one_rhythm_only = df_statement_counts[
    (df_statement_counts["total_statements"] == 1) & (df_statement_counts["n_rhythm_statements"] == 1)
]

n_one_rhythm_only = len(df_one_rhythm_only)
print(f"Number of records with ONLY 1 statement which is a rhythm statement: {n_one_rhythm_only}")

# If you want to see the scp_code for each, we can merge back with df_scp_breakdown:
df_one_rhythm_codes = df_scp_breakdown.merge(df_one_rhythm_only, on="record_id")
print("\nSample of those records (scp_code and flags):")
display(df_one_rhythm_codes.head(10))

# %% Phase 3: Cross-Domain Generalization (PTB-XL)
print("\n" + "="*70)
print("Phase 3: Cross-Domain Generalization with PTB-XL")
print("="*70)