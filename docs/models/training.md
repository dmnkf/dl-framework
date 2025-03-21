# Model Training

This guide explains how to train models using our PyTorch Lightning-based training framework.

## Overview

The training system consists of several key components:

1. **Lightning Modules**: Models implemented as PyTorch Lightning modules
2. **Training Script**: Central `train.py` script for experiment execution
3. **Docker Environment**: Containerized training environment
4. **Slurm Integration**: Scripts for cluster training

## Model Implementation

All models should be implemented as [PyTorch Lightning modules](https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html).

### Key Components

1. **Model Definition**: Inherit from `LightningModule`
2. **Training Step**: Implement `training_step`
3. **Validation Step**: Implement `validation_step`
4. **Test Step**: Implement `test_step`
5. **Configure Optimizers**: Implement `configure_optimizers`

```python
from lightning import LightningModule
import torch.nn as nn

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(...)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
```

## Training Script

The main training script (`src/train.py`) handles:

1. Configuration management via Hydra
2. DataModule initialization
3. Model instantiation
4. Logger and callback setup
5. Training execution

Training is recommended to be done in conjuction with [experiment management](../getting-started/configuration.md#experiment-configs).

### Basic Usage

```bash
# Train with experiment configuration
rye run train experiment=experiment_name
```

### Configuration

Training configurations are managed by Hydra and stored in `configs/`:

```yaml
# configs/train.yaml
defaults:
  - model: base_model
  - data: ptbxl
  - trainer: default
  - callbacks: default
  - logger: wandb

model:
  lr: 0.001
  hidden_size: 128

data:
  batch_size: 64
  num_workers: 4

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
```

## Training Environments

### Local Development

For local development and testing, you can run training directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py
```

### Docker Environment

For reproducible training, use the provided Docker environment:

```bash
# Build the container
cd docker
./build.sh

# Run training in container
docker run --gpus all ipoleprojection/projection:latest python src/train.py
```

## Cluster Training (Slurm)

For large-scale training on a Slurm cluster, use the provided scripts in `scripts/slurm/`:

### Basic Job Submission

```bash
# Submit a training job
sbatch scripts/slurm/train.sh -o "experiment=experiment_name"
```

### Resource Configuration

The Slurm scripts include default resource configurations:

```bash
#SBATCH -t 1-5:00:00     # Run time (days-hours:minutes:seconds)
#SBATCH -p performance   # Partition (queue)
#SBATCH --gpus=1        # Number of GPUs
#SBATCH --cpus-per-gpu=4 # CPUs per GPU
#SBATCH --mem=32G       # Memory per node
```

Modify these settings in `scripts/slurm/train.sh` based on your needs.

### Job Management

```bash
# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job logs
tail -f logs/projection_train_<job_id>.out
tail -f logs/projection_train_<job_id>.err
```

### Singularity Container

The training jobs run within a Singularity container to ensure reproducibility and consistent environments across cluster nodes. The container setup includes:

1. **Container Location**: Uses `$HOME/projection_latest.sif` as the container image
2. **Mounted Directories**:
      - Workspace: Job-specific workspace at `/tmp/projection_${SLURM_JOB_ID}` → `/workspace`
      - Data: Project data from `$DATA_PATH` → `/workspace/data`

3. **Environment Configuration**:
   ```bash
   SINGULARITYENV_LC_ALL=C.UTF-8
   SINGULARITYENV_LANG=C.UTF-8
   SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
   ```

4. **GPU Support**: Container runs with `--nv` flag for NVIDIA GPU access

To update the container image, use the provided script:
```bash
scripts/slurm/pull_latest.sh
```

This containerized approach ensures consistent software environments, proper isolation, and reproducible experiments across different compute nodes in the cluster.

## AWS SageMaker Training

> **Note**: The AWS SageMaker integration is currently in beta and serves as a proof of concept. While the setup provides a solid foundation for cloud training, it has not undergone extensive testing and can encounter issues.

The framework supports training on AWS SageMaker using custom ECR (Elastic Container Registry) images. This setup provides scalable cloud training with managed infrastructure.

### Prerequisites

1. **AWS CLI Setup**:
   ```bash
   # Configure AWS CLI with your credentials
   aws configure
   
   # Login to Amazon ECR
   aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account>.dkr.ecr.<your-region>.amazonaws.com
   ```

2. **Docker Image**:
   Base image: PyTorch training container from AWS Deep Learning Containers
   Location: `docker/aws/Dockerfile`
   Custom additions:
     - Rye package management
     - SageMaker directory structure
     - NVIDIA CUDA support

### Directory Structure
```
docker/aws/
  └── Dockerfile           # SageMaker-compatible container definition

scripts/aws/
  ├── entrypoint.sh       # Container entry point
  ├── sm_entrypoint.py    # Hydra configuration adapter
  ├── train.py            # SageMaker training script
  └── test.py            # Test script for validation
```

### Configuration

The training job uses Hydra configurations with SageMaker-specific adaptations:

1. **Hyperparameters**: Passed as SageMaker training job parameters are automatically translated to Hydra format
2. **Environment**: 
   Uses standard SageMaker paths:
      - `/opt/ml/input/data/training`: Training data
      - `/opt/ml/model`: Model artifacts
      - `/opt/ml/output`: Training outputs

### Usage

1. **Build and Push Image**:
   ```bash
   # From project root
   docker build -t <your-repo>/projection:latest -f docker/aws/Dockerfile .
   docker push <your-repo>/projection:latest
   ```

2. **Launch Training**:
   Use the AWS SageMaker SDK or console to launch a training job with:
      - Training image: Your pushed ECR image
      - Entry point: `/opt/ml/code/entrypoint.sh`
      - Hyperparameters: Passed as standard SageMaker hyperparameters

The SageMaker setup automatically handles infrastructure provisioning, data transfer, and artifact management while maintaining compatibility with the local training workflow through Hydra configurations.