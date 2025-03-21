 # Preprocessing

The `preprocessing/main.py` script is designed to preprocess medical data for downstream tasks. Specifically, it handles preprocessing of ECG (Electrocardiogram) signals and CMR (Cardiac Magnetic Resonance) images. The code allows you to configure the type of preprocessing you want, including data partitioning and normalization, which are essential steps in preparing medical datasets for machine learning.

Both modalities of data can be split into training, validation, and test sets, using strategies like random splitting, stratified splitting, and k-fold cross-validation.

### What Each Modality's Preprocessing Entails

- **ECG Signals**: The preprocessing includes:

  - Adjusting the sequence length.
  - Applying normalization with the following available modes:
    - `sample_wise`
    - `channel_wise`
    - `group_wise`

- **CMR Images**: The preprocessing includes:

  - Extracting 2D slices from 4D image data (time, height, width, slices).
  - Resizing images to a target dimension to maintain uniformity.
  - Optionally applying intensity normalization to standardize the images.

## Usage

To run the preprocessing script, you need to specify the data modality (either `ecg` or `cmr`) along with appropriate parameters. Below are some examples on how to use the script effectively.

### General Command Structure

```sh
python src/data/preprocessing/main.py <modality> [options]
```

or using the `rye` script defined in `pyproject.toml`:

```sh
rye run preprocess <modality> [options]
```

The `<modality>` can be either `ecg` or `cmr`.

### Command-Line Arguments

#### Common Arguments

These arguments apply to both `ecg` and `cmr` modalities:

- `--data_root` (required): Root directory containing the raw data.
- `--datasets`: Specific datasets to process (optional; if not provided, all available datasets will be processed).
- `--split_type` (default: `random`): Split type, options include `random`, `stratified`, `kfold`, `stratified_kfold`.
- `--val_size` (default: `0.1`): Fraction of data used for validation.
- `--test_size` (default: `0.1`): Fraction of data used for testing.
- `--random_seed` (default: `42`): Random seed for reproducibility.
- `--n_splits` (default: `1`): Number of splits for k-fold cross-validation.
- `--max_workers` (default: `None`): Number of workers for parallel processing. If not provided, processing will be done sequentially.

#### ECG-Specific Arguments

- `--sequence_length` (default: `5000`): Desired sequence length for ECG signals.
- `--normalize_mode` (default: `group_wise`): Normalization mode (`sample_wise`, `channel_wise`, or `group_wise`).

#### CMR-Specific Arguments

- `--image_size` (default: `256`): Target size for image resizing.
- `--normalize`: Apply intensity normalization to CMR images.

## Examples

### ECG Preprocessing

To preprocess ECG data using group-wise normalization and split the data randomly into train, validation, and test sets:

```sh
python src/data/preprocessing/main.py ecg --data_root /path/to/data --split_type random --sequence_length 5000 --normalize_mode group_wise
```

### CMR Preprocessing

To preprocess CMR images, resizing to 256x256, and applying intensity normalization:

```sh
python src/data/preprocessing/main.py cmr --data_root /path/to/data --image_size 256 --normalize
```

## Preprocessing Flow

The main function of the script (`main()`) handles:

1. Parsing command-line arguments.
2. Creating a data split configuration (`PartitioningConfig`).
3. Selecting and processing the datasets for the chosen modality.

The script relies on custom preprocessor classes (`ECGPreprocessor`, `CMRPreprocessor`), which extend a base preprocessor (`BasePreprocessor`) that defines common preprocessing operations such as:

- **Preparing interim directories** for processed data.
- **Creating splits** for train/validation/test.
- **Loading and saving preprocessed records**.

## Error Handling

The script includes logging to help troubleshoot issues. Any problems encountered during dataset processing are logged, and an appropriate error message is displayed. For instance, if the raw data path doesn't exist or a dataset is incorrectly specified, the script will log the issue and halt execution.

## Output

The processed datasets are saved to an interim directory, structured to include the dataset information, preprocessing metadata, and data splits. Each sample is saved as a `.pt` file (PyTorch Tensor) for further use in training.

## Logging

The script logs progress and issues using Python's logging library. You will see informative messages during the preprocessing that include:

- Dataset being processed.
- Preprocessing progress using `tqdm`.
- Any errors that occur.
