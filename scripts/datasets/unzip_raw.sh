#!/bin/bash

# Define the base directory using absolute paths
BASE_DIR="$(realpath data/raw-zips)"
TARGET_BASE_DIR="$(realpath data/raw)"

# Confirm paths with the user
echo "Base Directory: $BASE_DIR"
echo "Target Directory: $TARGET_BASE_DIR"
echo "Proceed with these paths? (yes/no)"
read confirmation
if [[ "$confirmation" != "yes" ]]; then
    echo "Operation aborted."
    exit 1
fi

# Check for force flag
FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
    echo "Force mode enabled: Clearing $TARGET_BASE_DIR first."
    rm -rf "$TARGET_BASE_DIR"
fi

# Find all zip files in the raw-zips directory and its subdirectories
find "$BASE_DIR" -type f -name "*.zip" | while read zipfile; do
    # Extract the directory containing the zip file relative to BASE_DIR
    relative_dir=$(dirname "${zipfile#$BASE_DIR/}")
    
    # Extract the filename without extension
    zip_name=$(basename "$zipfile" .zip)
    
    # Construct the extraction directory inside the target base directory
    extract_dir="$TARGET_BASE_DIR/$relative_dir/$zip_name"
    
    # Ensure extraction directory is an absolute path
    # extract_dir="$(realpath --canonicalize-missing "$extract_dir")"
    
    # Check if the extraction directory already exists (unless force mode is enabled)
    if [ -d "$extract_dir" ] && [ "$FORCE" = false ]; then
        echo "Warning: $extract_dir already exists. Skipping extraction."
        continue
    fi
    
    mkdir -p "$extract_dir"
    
    # Unzip the contents into the created folder
    unzip -o "$zipfile" -d "$extract_dir"
    
    echo "Extracted $zipfile to $extract_dir"
done
