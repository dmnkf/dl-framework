# Docker Development Environment

This guide explains how to use Docker for development and deployment in the Aether project. Our Docker setup provides a consistent environment with CUDA support, development tools, and the Rye package manager.

## Container Architecture

### Base Image

We use `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04` as our base image for:

- CUDA 11.7.1 and cuDNN 8 support
- Development tools compatibility
- Ubuntu 22.04 LTS stability

### Key Components

1. **Environment Configuration**
   ```dockerfile
   ENV DEBIAN_FRONTEND=noninteractive \
       LANG=C.UTF-8 \
       RYE_HOME=/app/.rye \
       LC_ALL=C.UTF-8
   ```

2. **Development Tools**
      - Git for version control
      - Build essentials for compilation
      - Rye package manager for Python dependencies

3. **Python Environment**
      - Managed by Rye package manager
      - Dependencies from `pyproject.toml`
      - Locked versions in `requirements.lock`

## Building Images

### Using the Build Script

The `build.sh` script provides a flexible way to build and manage Docker images:

```bash
# Basic build
./docker/build.sh

# Build with custom name and tag
./docker/build.sh --name myproject --tag v1.0

# Build and convert to Singularity
./docker/build.sh --singularity

# Build and push to registry
./docker/build.sh --push --registry your.registry.com/username
```

### Build Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Image name | "projection" |
| `--tag` | Image tag | "latest" |
| `--singularity` | Convert to Singularity | false |
| `--push` | Push to registry | false |
| `--registry` | Registry URL | "" |

## Development Workflow

### Local Development

1. **Build the Image**
   ```bash
   ./docker/build.sh
   ```

2. **Run Development Container**
   ```bash
   docker run --gpus all -it \
       -v $(pwd):/workspace \
       -p 8888:8888 \  # for Jupyter
       projection:latest
   ```

3. **Working with the Container**
      - Code changes in host are reflected in container
      - Python environment is pre-configured
      - GPU support is enabled

### Best Practices

1. **Volume Mounting**
      - Mount your project directory as `/workspace`
      - Consider mounting data directories separately
      - Use named volumes for persistent storage

2. **Environment Variables**
   ```bash
   docker run -it \
       -e CUDA_VISIBLE_DEVICES=0,1 \
       -e WANDB_API_KEY=your_key \
       projection:latest
   ```

3. **Resource Management**
   ```bash
   docker run -it \
       --gpus '"device=0,1"' \
       --cpus=4 \
       --memory=16g \
       projection:latest
   ```

## Cluster Deployment

### Converting to Singularity

1. **Build and Convert**
   ```bash
   ./docker/build.sh --singularity
   ```

2. **Transfer to Cluster**
   ```bash
   scp projection_latest.sif username@cluster:/path/to/home/
   ```

3. **Running on Cluster**
   ```bash
   singularity run --nv projection_latest.sif
   ```

### SLURM Integration

For SLURM-based clusters, create job scripts like:

```bash
#!/bin/bash
#SBATCH --job-name=aether_training
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

singularity run --nv \
    -B /path/to/data:/data \
    projection_latest.sif \
    python train.py
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Access**
      - Ensure NVIDIA drivers are installed
      - Use `nvidia-smi` to verify GPU access
      - Check `--gpus` flag in docker run

2. **Volume Permissions**
   ```bash
   # Fix permission issues
   docker run -it \
       -u $(id -u):$(id -g) \
       -v $(pwd):/workspace \
       projection:latest
   ```

3. **Package Installation**
      - Update `pyproject.toml` and rebuild
      - Use `rye add package_name` inside container
      - Check Rye environment activation

### Debugging Tips

1. **Container Inspection**
   ```bash
   # Enter running container
   docker exec -it container_name bash
   
   # Check logs
   docker logs container_name
   ```

2. **Build Issues**
   ```bash
   # Build with verbose output
   docker build --progress=plain -t projection:latest .
   ```