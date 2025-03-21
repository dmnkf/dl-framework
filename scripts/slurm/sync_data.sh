#!/bin/bash

# Default values
SOURCE_DIR=""
TARGET_DIR=""
SSH_CONFIG="i4ds_slurm"
DRY_RUN=false
EXCLUDE_FILE=""
DIRECTION="push"  # Default to pushing to cluster

# Help message
usage() {
    echo "Usage: $0 -s <source_dir> -t <target_dir> [-n] [-e <exclude_file>] [-h <ssh_config>] [-d <direction>]"
    echo "  -s: Source directory to sync from"
    echo "  -t: Target directory to sync to"
    echo "  -n: Dry run (show what would be transferred)"
    echo "  -e: Path to exclude file (optional)"
    echo "  -h: SSH config entry name (default: i4ds_slurm)"
    echo "  -d: Direction of sync: push or pull (default: push)"
    echo "      push: local -> cluster"
    echo "      pull: cluster -> local"
    exit 1
}

# Parse command line arguments
while getopts "s:t:ne:h:d:" opt; do
    case $opt in
        s) SOURCE_DIR="$OPTARG" ;;
        t) TARGET_DIR="$OPTARG" ;;
        n) DRY_RUN=true ;;
        e) EXCLUDE_FILE="$OPTARG" ;;
        h) SSH_CONFIG="$OPTARG" ;;
        d) DIRECTION="$OPTARG" ;;
        ?) usage ;;
    esac
done

# Check required arguments
if [ -z "$SOURCE_DIR" ] || [ -z "$TARGET_DIR" ]; then
    echo "Error: Source and target directories are required"
    usage
fi

# Validate direction
if [ "$DIRECTION" != "push" ] && [ "$DIRECTION" != "pull" ]; then
    echo "Error: Direction must be either 'push' or 'pull'"
    usage
fi

# Verify SSH config exists
if ! ssh -G "$SSH_CONFIG" >/dev/null 2>&1; then
    echo "Error: SSH config entry '$SSH_CONFIG' not found"
    exit 1
fi

# Construct rsync command
RSYNC_CMD="rsync -avz --progress -e ssh"

# Add dry run flag if specified
if [ "$DRY_RUN" = true ]; then
    RSYNC_CMD="$RSYNC_CMD --dry-run"
fi

# Add exclude file if specified
if [ -n "$EXCLUDE_FILE" ]; then
    if [ -f "$EXCLUDE_FILE" ]; then
        RSYNC_CMD="$RSYNC_CMD --exclude-from=$EXCLUDE_FILE"
    else
        echo "Warning: Exclude file $EXCLUDE_FILE not found"
    fi
fi

# Ensure source directory ends with /
SOURCE_DIR="${SOURCE_DIR%/}/"

# Construct source and target paths based on direction
if [ "$DIRECTION" = "push" ]; then
    FROM_PATH="$SOURCE_DIR"
    TO_PATH="$SSH_CONFIG:$TARGET_DIR"
    echo "Pushing from local ($SOURCE_DIR) to cluster ($SSH_CONFIG:$TARGET_DIR)"
else
    FROM_PATH="$SSH_CONFIG:$SOURCE_DIR"
    TO_PATH="$TARGET_DIR"
    echo "Pulling from cluster ($SSH_CONFIG:$SOURCE_DIR) to local ($TARGET_DIR)"
fi

# Execute rsync
$RSYNC_CMD "$FROM_PATH" "$TO_PATH"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Sync completed successfully"
else
    echo "Sync failed with exit code $exit_code"
fi

exit $exit_code 