# Docker Setup

This directory contains Docker-related files for building and managing containers for the project.

## Directory Structure

```
docker/
├── Dockerfile          # Main Dockerfile for the project
├── build.sh           # Helper script for building Docker/Singularity images
└── README.md          # This file
```

## Base Image

We use `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04` as our base image because:
- Includes CUDA 11.7.1 and cuDNN 8
- Provides development tools needed for some Python packages
- Compatible with our PyTorch requirements

## Key Features in the Dockerfile

1. **Environment Setup**:
   - Configures `DEBIAN_FRONTEND` and locale settings (`LANG`, `LC_ALL`).
   - Defines `RYE_HOME` for the Rye package manager.

2. **Package Installation**:
   - Installs essential development tools like `git`, `curl`, `build-essential`, and `software-properties-common`.

3. **Rye Package Manager**:
   - Installs Rye using the official installation script.
   - Syncs dependencies from `pyproject.toml` and `requirements.lock` files.

4. **Optimizations**:
   - Removes unnecessary apt cache files after installation.
   - Uses a clean working directory (`/workspace`) for development.

5. **Custom Entry Point**:
   - Configures the container to activate the Rye environment and its virtual environment upon starting.

## Building Images

The `build.sh` script provides a convenient way to build Docker images and optionally convert them to Singularity.

### Basic Usage

```bash
# Build Docker image
./docker/build.sh

# Build and convert to Singularity
./docker/build.sh --singularity

# Build with custom name and tag
./docker/build.sh --name myproject --tag v1.0

# Build and push to registry
./docker/build.sh --push --registry your.registry.com/username
```

### All Options

- `--name`: Set image name (default: "projection")
- `--tag`: Set image tag (default: "latest")
- `--singularity`: Convert to Singularity after building
- `--push`: Push to Docker registry
- `--registry`: Specify registry URL for pushing

## Development Workflow

1. **Local Development**:
   ```bash
   # Build image
   ./docker/build.sh

   # Run container for development
   docker run --gpus all -it \
       -v $(pwd):/workspace \
       projection:latest
   ```

2. **Cluster Deployment**:
   ```bash
   # Build and convert to Singularity
   ./docker/build.sh --singularity

   # Copy to cluster
   scp projection_latest.sif username@cluster:/path/to/home/
   ```

   Read more about the cluster related setup in the [SLURM README](../scripts/slurm/README.md).

