# Working with Datasets

## Overview

This guide explains how to work with datasets in the project. The system is designed to handle multiple modalities (ECG, CMR) through a unified pipeline that consists of three main stages:

1. **Raw Data Handling**: Loading and validating original data files
2. **Preprocessing**: Converting data into standardized tensor format
3. **Unified Access**: Bringing everything together for analysis and training

## Data Organization

The project follows a consistent directory structure across all datasets:

```
data/
├── raw/                    # Original dataset files
│   ├── cmr/                    # Cardiac MRI datasets
│   │   └── acdc/               # ACDC dataset files
│   └── ecg/                    # ECG datasets
│       └── ptbxl/              # PTB-XL dataset files
├── interim/                # Preprocessed tensors (*.pt files)
│   ├── acdc/                   # Preprocessed ACDC records
│   └── ptbxl/                  # Preprocessed PTB-XL records
├── processed/              # Final dataset artifacts
│   ├── acdc/                   # ACDC splits and metadata
│   │   ├── splits.json         # Train/val/test splits
│   │   └── metadata.db         # Record metadata
│   └── ptbxl/                  # PTB-XL splits and metadata
└── embeddings/             # Pre-computed embeddings (optional)
    ├── type1/                  # Embeddings from model 1
    └── type2/                  # Embeddings from model 2
```

## Dataset Components

### 1. Raw Dataset Handlers

Raw dataset handlers provide the interface to original data files. They handle:

- Data loading and validation
- Metadata extraction
- Label processing
- Format standardization

The base classes that all handlers must implement:

::: src.data.raw.data
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [RawDataset, RawRecord]
        show_docstring_description: true

---

Dataset handlers are registered using:

::: src.data.raw.registry
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [RawDatasetRegistry]
        show_docstring_description: true

### 2. Unified Dataset System

The unified dataset system brings everything together, providing:

- Access to raw, preprocessed, and embedded data
- Automatic data integrity validation
- Efficient caching
- Comprehensive metadata management

::: src.data.unified
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [UnifiedDataset, UnifiedRecord]
        show_docstring_description: true

## Implementation Guide

Now that you have a basic understanding of the unified dataset system, let's go through the implementation steps for creating a new dataset handler given some raw data.

### Creating a New Dataset Handler

**1. Choose the Appropriate Location:**

- ECG handlers: `src/data/raw/ecg/<dataset_name>.py`
- CMR handlers: `src/data/raw/cmr/<dataset_name>.py`

**2. Implement Required Methods:**

All handlers must implement the `RawDataset` abstract base class, which defines the structure and contents of the raw data.

::: src.data.raw.data
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [RawDataset]
        show_docstring_description: true


The `RawRecord` object is defined along the same lines as the `RawDataset` class, and is used to represent a single record in the dataset. It contains the raw data and metadata for that record ready for preprocessing.

::: src.data.raw.data
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [RawRecord]
        show_docstring_description: true

**3. Register the Dataset Handler:**

In order to use your new dataset handler, you need to register it in the `src/data/__init__.py` module.

```python
# Import datasets in order to register them
import src.data.raw.cmr.acdc
import src.data.raw.ecg.arrhythmia
import src.data.raw.ecg.grouped_arrhythmia
import src.data.raw.ecg.shandong
import src.data.raw.ecg.ptb_xl
```

### Example Implementations of Dataset Handlers

::: src.data.raw.ecg.ptb_xl
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [PTBXL]
        show_docstring_description: true

::: src.data.raw.cmr.acdc
    options:
        show_root_heading: true
        heading_level: 4
        show_source: true
        members: [ACDC]
        show_docstring_description: true

### Using the Unified Dataset

In this section, we will see how to use the `UnifiedDataset` class to load and preprocess data from a dataset.

The `UnifiedDataset` can easily be initialized using the `data_root`, `modality` and `dataset_key` arguments. It will then load the dataset's metadata and provide methods for preprocessing and embedding the data.

```python
from pathlib import Path
from src.data.dataset import DatasetModality
from src.data.unified import UnifiedDataset

# Initialize dataset
dataset = UnifiedDataset(
    data_root=Path("data"),
    modality=DatasetModality.ECG,
    dataset_key="ptbxl"
)

# Access data
record = dataset["patient_001"]
raw_data = record.raw_record.data
preprocessed = record.preprocessed_record
embeddings = record.embeddings

# Get splits and metadata
splits = dataset.get_splits() if dataset.has_splits() else None
metadata_fields = dataset.available_metadata_fields()

# Verify integrity
dataset.verify_integrity()
```

## Next Steps

- See [Data Preprocessing](preprocessing.md) for details on data preprocessing
- Check [Training Models](../models/training.md) for using your dataset in training
- Review [Data Management](data-management.md) for handling data versions
