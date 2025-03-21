#!/bin/bash

# Default values
DATA_ROOT="data"
MAX_WORKERS=5
FORCE_RESTART=false
MODALITY=""
DATASET=""
SPLIT_TYPE="random"

# Help message
usage() {
    echo "Usage: $0 -m <modality> -d <dataset> [-r <data_root>] [-w <max_workers>] [-f] [-s <split_type>]"
    echo "  -m: Modality type (e.g., cmr, ecg)"
    echo "  -d: Dataset name (e.g., acdc for cmr)"
    echo "  -r: Data root directory (default: data)"
    echo "  -w: Maximum number of workers (default: 5)"
    echo "  -f: Force restart preprocessing"
    echo "  -s: Split type for partitioning (default: random)"
    exit 1
}

# Parse command line arguments
while getopts "m:d:r:w:fs:" opt; do
    case $opt in
        m) MODALITY="$OPTARG" ;;
        d) DATASET="$OPTARG" ;;
        r) DATA_ROOT="$OPTARG" ;;
        w) MAX_WORKERS="$OPTARG" ;;
        f) FORCE_RESTART=true ;;
        s) SPLIT_TYPE="$OPTARG" ;;
        ?) usage ;;
    esac
done

# Check required arguments
if [ -z "$MODALITY" ] || [ -z "$DATASET" ]; then
    echo "Error: Both modality and dataset are required"
    usage
fi

# Construct preprocessing command
PREPROCESS_CMD="rye run preprocess $MODALITY --data_root $DATA_ROOT --max_workers=$MAX_WORKERS --datasets $DATASET"
if [ "$FORCE_RESTART" = true ]; then
    PREPROCESS_CMD="$PREPROCESS_CMD --force_restart"
fi

# Execute preprocessing
echo "Step 1: Running preprocessing for $MODALITY modality (dataset: $DATASET)..."
echo "Command: $PREPROCESS_CMD"
$PREPROCESS_CMD
preprocess_exit=$?

if [ $preprocess_exit -ne 0 ]; then
    echo "Preprocessing failed with exit code $preprocess_exit"
    exit $preprocess_exit
fi

echo "Preprocessing completed successfully"

# Construct paths for partitioning
DATA_DIR="$DATA_ROOT/interim/$DATASET"
OUTPUT_DIR="$DATA_ROOT/processed/$DATASET"

# Construct partitioning command
PARTITION_CMD="rye run partition --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --split_type $SPLIT_TYPE"

# Execute partitioning
echo -e "\nStep 2: Running partitioning..."
echo "Command: $PARTITION_CMD"
$PARTITION_CMD
partition_exit=$?

if [ $partition_exit -eq 0 ]; then
    echo "Partitioning completed successfully"
else
    echo "Partitioning failed with exit code $partition_exit"
fi

exit $partition_exit 