#!/bin/bash

#SBATCH -t 1-5:00:00
#SBATCH -p performance
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --job-name=projection_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

overrides=""

while getopts e:o: flag; do
    case "${flag}" in
        o) overrides=${OPTARG};;
    esac
done

echo "Overrides: $overrides"
export overrides

PROJECT_ROOT="$HOME/projection"
DATA_PATH="$HOME/projection_data"
CONTAINER_PATH="$HOME/projection_latest.sif"
DESTINATION_DIR="$HOME/ckpt_backups"

mkdir -p logs

echo "Setting up workspace..."
WORKSPACE_DIR="/tmp/projection_${SLURM_JOB_ID}"
echo "Creating workspace copy at: $WORKSPACE_DIR"
cp -r "$PROJECT_ROOT" "$WORKSPACE_DIR"

cd "$WORKSPACE_DIR" || { echo "Error: Could not navigate to $WORKSPACE_DIR."; exit 1; }

if [ -f .env ]; then
    export $(cat .env | grep WANDB_API_KEY | xargs)
fi

SINGULARITYENV_LC_ALL=C.UTF-8 \
SINGULARITYENV_LANG=C.UTF-8 \
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY \
SINGULARITYENV_SLURM_JOB_NAME=$SLURM_JOB_NAME \
SINGULARITYENV_SLURM_JOB_ID=$SLURM_JOB_ID \
singularity exec \
    -B "$WORKSPACE_DIR":/workspace \
    -B "$DATA_PATH":/workspace/data \
    --nv "$CONTAINER_PATH" \
    bash -c "
        export LC_ALL=C.UTF-8 && \
        export LANG=C.UTF-8 && \
        source /app/.rye/env && \
        source /app/.rye/global/.venv/bin/activate && \
        cd /workspace && \
        python3 src/train.py $overrides
    "

RUN_SPECIFIC_DIR="$DESTINATION_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_SPECIFIC_DIR"

echo "Copying entire logs directory to $RUN_SPECIFIC_DIR..."
cp -r logs "$RUN_SPECIFIC_DIR/"

echo "Cleaning up workspace..."
rm -rf "$WORKSPACE_DIR"
echo "Training finished"