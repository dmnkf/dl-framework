from src.data.store.metadata_store import MetadataStore
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=1.2)

np.random.seed(42)

MAX_SAMPLES = 500

project_root = Path().absolute().parent.parent
data_root = project_root / "data"
pre_trained_embeddings_root = project_root / "embeddings" / "baseline"
fine_tuned_embeddings_root = project_root / "embeddings" / "fine_tuned"

ecg_embeddings_file = "arrhythmia.pt"
cmr_embeddings_file = "acdc.pt"


def calculate_bmi(row):
    """Calculates BMI given height (in cm) and weight (in kg)."""
    try:
        weight = float(row["Weight"])
        height = float(row["Height"]) / 100  # Convert cm to meters
        if height > 0:
            return weight / (height**2)
    except (ValueError, TypeError):
        return np.nan
    return np.nan


def subsample_data_by_ids(embeddings_data, sample_ids):
    """
    Subsamples data using provided sample IDs.
    Returns embeddings, labels, and metadata for matching samples.
    """
    # Create a mapping of ID to index
    id_to_idx = {entry["idx"]: i for i, entry in enumerate(embeddings_data)}

    # Get indices for the requested sample IDs
    indices = [id_to_idx[id_] for id_ in sample_ids if id_ in id_to_idx]

    # Extract embeddings and other data
    embeddings = np.array([embeddings_data[i]["embeddings"].numpy() for i in indices])
    embedding_ids = [embeddings_data[i]["idx"] for i in indices]

    return embeddings, embedding_ids, indices


def load_data(modality, dataset_key=None):
    """
    Load and prepare data for analysis.

    Args:
        modality: str, either 'ECG' or 'CMR'
        dataset_key: str, optional, specific dataset key (e.g., 'arrhythmia' or 'acdc')
    """
    # Set default dataset key if not provided
    if dataset_key is None:
        dataset_key = "arrhythmia" if modality.lower() == "ecg" else "acdc"

    # Configure paths based on modality
    if modality.lower() == "ecg":
        pre_embeddings_path = pre_trained_embeddings_root / ecg_embeddings_file
        fine_embeddings_path = fine_tuned_embeddings_root / ecg_embeddings_file
        metadata_path = data_root / "processed" / dataset_key
        is_ecg = True
    elif modality.lower() == "cmr":
        pre_embeddings_path = pre_trained_embeddings_root / cmr_embeddings_file
        fine_embeddings_path = fine_tuned_embeddings_root / cmr_embeddings_file
        metadata_path = data_root / "processed" / dataset_key
        is_ecg = False
    else:
        raise ValueError("Invalid modality. Choose 'ECG' or 'CMR'.")

    # Load raw data
    pre_data = torch.load(pre_embeddings_path)
    fine_data = torch.load(fine_embeddings_path)

    # Get all sample IDs
    all_ids = list(set(entry["idx"] for entry in pre_data))

    # Subsample IDs if necessary
    if len(all_ids) > MAX_SAMPLES:
        sample_ids = np.random.choice(all_ids, MAX_SAMPLES, replace=False)
    else:
        sample_ids = all_ids

    # Subsample both datasets using the same IDs
    embeddings_pre, ids_pre, indices_pre = subsample_data_by_ids(pre_data, sample_ids)
    embeddings_fine, ids_fine, indices_fine = subsample_data_by_ids(
        fine_data, sample_ids
    )

    # Verify that samples are the same
    assert (
        ids_pre == ids_fine
    ), "Sample IDs do not match between pre-trained and fine-tuned embeddings."
    assert len(ids_pre) == len(
        sample_ids
    ), "Mismatch in sample count after subsampling."

    # Load metadata
    metadata_store = MetadataStore(dataset_key, metadata_path)
    metadata = metadata_store.get_batch(ids_pre)

    # Process metadata and labels
    metadata_list = []
    labels = []

    for record_id in ids_pre:
        record_metadata = metadata.get(record_id)
        if not record_metadata:
            continue

        if is_ecg:
            labels_grouped = record_metadata.get("labels_group", [])
            primary_label = labels_grouped[0] if labels_grouped else "Unknown"
            labels.append(primary_label)

            metadata_list.append(
                {
                    "id": record_id,
                    "Age": record_metadata.get("age"),
                    "Sex": "Male" if record_metadata.get("is_male") else "Female",
                    "Labels": labels_grouped,
                    "Label_Codes": record_metadata.get("labels_int_codes", []),
                    "Primary_Label": primary_label,
                }
            )
        else:
            label = next(
                entry["targets"][0] for entry in pre_data if entry["idx"] == record_id
            )
            labels.append(label)

            metadata_list.append(
                {
                    "id": record_id,
                    "Height": record_metadata.get("height", "Unknown"),
                    "Weight": record_metadata.get("weight", "Unknown"),
                    "Labels": label,
                    "Primary_Label": label,
                }
            )

    metadata_df = pd.DataFrame(metadata_list)

    # Add derived columns
    if is_ecg:
        metadata_df["Age"] = metadata_df["Age"].fillna("Unknown")
        metadata_df["Sex"] = metadata_df["Sex"].fillna("Unknown")
        metadata_df["Age_Group"] = (
            pd.cut(
                pd.to_numeric(metadata_df["Age"], errors="coerce"),
                bins=[0, 30, 50, 70, np.inf],
                labels=["<30", "30-50", "50-70", "70+"],
                right=False,
            )
            .astype(str)
            .fillna("Unknown")
        )
    else:
        metadata_df["BMI"] = metadata_df.apply(calculate_bmi, axis=1)
        metadata_df["BMI_Group"] = (
            pd.cut(
                metadata_df["BMI"],
                bins=[0, 18.5, 25, 30, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                right=False,
            )
            .astype(str)
            .fillna("Unknown")
        )

    return embeddings_pre, labels, metadata_df, embeddings_fine, labels, metadata_df
