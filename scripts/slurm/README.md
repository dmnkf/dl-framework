# i4DS Slurm Cluster Training Guide

This guide explains how to run training jobs on the cluster.

## Table of Contents

- [i4DS Slurm Cluster Training Guide](#i4ds-slurm-cluster-training-guide)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Cluster Access and Setup](#cluster-access-and-setup)
  - [Important Cluster Rules](#important-cluster-rules)
  - [How to run commands](#how-to-run-commands)
  - [Uploading Data](#uploading-data)
  - [Container Setup](#container-setup)
    - [Building the Container Locally](#building-the-container-locally)
    - [Pulling the Container from Docker Hub (Currently Recommended)](#pulling-the-container-from-docker-hub-currently-recommended)
    - [Pulling the Container from Gitlab Registry](#pulling-the-container-from-gitlab-registry)
  - [Update Script for Git and Singularity Image Management](#update-script-for-git-and-singularity-image-management)
  - [How to run experiments](#how-to-run-experiments)
    - [Configuration](#configuration)
    - [Basic Job Submission](#basic-job-submission)
    - [Resource Configuration](#resource-configuration)
  - [Monitoring Jobs](#monitoring-jobs)
    - [Basic Monitoring](#basic-monitoring)
    - [Advanced Monitoring](#advanced-monitoring)
  - [Data Management](#data-management)
  - [Partitions](#partitions)
  - [Directory Structure](#directory-structure)
  - [Troubleshooting](#troubleshooting)
    - [Additional Troubleshooting Tips](#additional-troubleshooting-tips)
  - [References](#references)


## Prerequisites

1. Access to a Slurm cluster with GPU support
2. Singularity/Apptainer installed on the cluster
3. Docker installed on your local machine (for building the container)

### Cluster Access and Setup
> From cluster documentation

1. Join the "slurm-gpu-cluster" Slack channel
2. Request login access from a Slurm manager
3. Provide your public SSH key (generate with `ssh-keygen -t ed25519 -C "your_email@example.com"`)
4. Connect via: `ssh -i [key_path] [username]@slurmlogin.cs.technik.fhnw.ch`

Note: The cluster is only accessible from FHNW-Eduroam or via VPN.

## Important Cluster Rules
> From cluster documentation

1. Never request more than 3 GPUs (preferably use 1)
2. Time limits:
   - Maximum job duration: 48 hours (`--time=48:00:00`)
   - Use minimal time needed to allow resource sharing
3. Never start jobs directly on nodes - always use Slurm
4. Your home directory (`/home2/username`) is shared across the cluster


## How to run commands

In order to run commands on the cluster, you can spawn an interactive shell using `srun`:

```bash
srun --performance performance --pty bash
```

This will give you an interactive shell on a compute node. You can then run your commands as needed.

To run a command directly on the cluster, you can use `srun`:

```bash
srun --performance performance echo "Hello, World!"
# or more generally
srun --performance performance <command>
```

## Uploading Data

1. **Small Datasets**:
   - Upload to your home directory: `/home2/username`
   - Copy to job workspace: `/tmp/projection_<job_id>/data/` (automatically done by the script)

2. **Large Datasets**:
   - Upload to `/mnt/nas05/data01` or `/mnt/nas05/data02`
   - Mount in job workspace: `/tmp/projection_<job_id>/data/`

Upload data to the cluster:
```bash
scp -rv <SOURCE_DIR> i4ds_slurm:/home2/<USERNAME>/<TARGET_DIR>
```

**Note:** As out current dataset setup is not yet large, we can use the home directory for now. Thus, we need to individually copy the data to our respective home directories.

## Container Setup

### Building the Container Locally

1. Build the Docker container locally using the provided build script. You can specify optional arguments to customize the build process. By default, the image is built with the name `projection` and tag `latest`:

```bash
bash ./docker/build.sh
```

The script will build the Docker image and tag it as `projection:latest`. You can customize the build process using the following optional arguments:
- `--name`: Specify a custom image name (default: `projection`).
- `--tag`: Specify a custom tag for the image (default: `latest`).
- `--singularity`: Convert the Docker image to a Singularity image.
- `--push`: Push the Docker image to a container registry.
- `--registry`: Specify the registry URL when pushing the image.

For example, to build the docker image and convert it to a Singularity image, you can run the following command:

```bash
bash ./docker/build.sh --singularity
```

This will build the Docker image and convert it to a Singularity image with the default name `projection_latest.sif`.

2. Copy the Singularity image to your cluster. You can use `scp` to copy the image to your cluster:

```bash
scp projection_latest.sif username@cluster:/path/to/home/
```

### Pulling the Container from Docker Hub (Currently Recommended)

Pull the container from Docker Hub:
```bash
singularity pull docker://ipoleprojection/projection:<TAG>
```

The file will be saved as `projection_<TAG>.sif`. Copy this to your cluster if needed.
```bash
scp projection_<TAG>.sif username@cluster:/path/to/home/
```

### Pulling the Container from Gitlab Registry

First login to the Gitlab registry:
```bash
singularity registry login --username <username> --password <token> docker://cr.gitlab.fhnw.ch
```

Then pull the container:
```bash
singularity pull docker://cr.gitlab.fhnw.ch/<username>/projection:<tag>
```

## Update Script for Git and Singularity Image Management

In order to consolidate the process of updating the Git repository and pulling the latest Singularity image, a utility script is provided. This script automates the process of updating the Git repository to the latest commit on the desired branch and pulling the latest Singularity image for the project.

Everything the script does can be done manually with the info provided in the previous sections. However, the script provides a convenient way to automate these steps.

The script handles the following tasks:
1. Updates the specified Git repository to the latest commit on the desired branch.
2. Pulls the latest Singularity image for the project from a specified registry.

The script is named `pull_latest.sh` and is located in the `scripts/slurm` directory. Here is how to use the script:

1. **Basic Command**:
   To use the script, specify the necessary parameters.
   ```bash
   sbatch pull_latest.sh -b <branch>
   ```

2. **Optional Arguments**:
   - `-r <registry>`: Specify the container registry URL. The default is `docker://ipoleprojection/projection`.
   - `-b <branch>`: Specify the branch to pull from Git. If not provided, the default is `main`. The branch name will automatically be converted to a valid tag format (e.g., replacing `/` with `-`).
   - `-t <tag>`: Specify the tag for the Singularity image. The default is `latest`. The tag will automatically be converted to a valid format (e.g., replacing `/` with `-`). If branch is provided, the tag will be set to the branch name.

3. **Example Command**:
   ```bash
   sbatch pull_latest.sh -t latest -r docker://myregistry/myproject -b feature/new-update
   ```

   This command:
   - Pulls the latest code from the `feature/new-update` branch.
   - Uses the tag `feature-new-update` for the Singularity image.
   - Fetches the Singularity image from the specified registry.

## How to run experiments

### Configuration

Configure paths in `train.sh`:
```bash
# In train.sh
PROJECT_ROOT="$HOME/projection"                    # Your project directory
DATA_PATH="$HOME/path/to/data"                     # Your data directory (currently in $HOME)
CONTAINER_PATH="$HOME/projection_latest.sif"       # Path to your container image. Note: Update this if you are using a different branch/tag as the image name will change
```

### Basic Job Submission

Submit a basic training job with the Hydra experiment name:

```bash
sbatch train.sh -o experiment=<experiment_name>
```

### Resource Configuration

The script includes default Slurm settings:
```bash
#SBATCH -t 1-5:00:00             # Time limit: 1 day, 5 hours
#SBATCH -p performance           # Partition (queue)
#SBATCH --gpus=1                 # Number of GPUs
#SBATCH --job-name=projection    # Job name
```

Override these when submitting:
```bash
# Request different resources
sbatch --time=2-0:00:00 --gpus=2 train.sh -o experiment=<experiment_name>
```

## Monitoring Jobs

### Basic Monitoring
```bash
# View your job queue
squeue -u $USER

# Check job logs in real-time
tail -f logs/projection_<job_id>.out

# Cancel a job
scancel <job_id>

# View detailed job information
scontrol show job <job_id>
```

### Advanced Monitoring
> From cluster documentation

```bash
# Detailed job view with GPU information
squeue -O 'JobID:.7,Partition:.12,UserName:.12,State:.7,TimeUsed:.10,NumCPUs:.4,tres-per-node:.13,tres-per-job:.12,NodeList:.8,Reason'

# Detailed job information including GPU allocation
scontrol show job [job_id] -d

# Online monitoring dashboard
URL: https://zagraf.it4ds.ch/d/R9OCYmTVz/slurm-cluster-i4ds
```

## Data Management
> From cluster documentation

Available storage locations:
- `/mnt/nas05/data01` (95TB) - For large datasets
- `/mnt/nas05/data02` (60TB) - Additional storage
- `/cluster/common` - Shared space for cluster users
- `/cluster/groups` - Group-specific storage
- `/home2/username` - Personal home directory (shared across cluster)

For sensitive data:
1. Copy to encrypted tar/gz on data01
2. Decrypt to `/cluster/group/projectname/`

## Partitions
> From cluster documentation

- `performance`: Default partition for GPU calculations
- `top6`: Access to fastest GPUs
- `CASonly`: Higher priority for CAS group users

## Directory Structure

Each job gets its own workspace:
```
/tmp/projection_<job_id>/
├── src/
├── configs/
├── data/ -> mounted from DATA_PATH
└── ... (other project files)
```

Logs are stored in:
```
$HOME/logs/
├── projection_<job_id>.out  # Standard output
└── projection_<job_id>.err  # Error output
```

## Troubleshooting

1. **Job Submission Issues**
   - Check partition availability: `sinfo`
   - Verify resource requests match partition limits
   - Test with shorter runtime in debug partition

2. **Container Issues**
   - Verify container path exists
   - Test container: `singularity shell --nv projection_latest.sif`
   - Check GPU access: `srun --pty nvidia-smi`

3. **Job Failures**
   - Check error logs: `logs/projection_<job_id>.err`
   - Verify all paths exist
   - Ensure container has necessary dependencies

### Additional Troubleshooting Tips
> From cluster documentation

1. For GPU issues:
   - Verify CUDA packages in container
   - Check GPU allocation with `nvidia-smi`
   - Ensure `--nv` flag with Singularity

2. For storage issues:
   - Check quota limits
   - Use scratch space for temporary files
   - Monitor disk usage with `df -h`

---

## References

- [i4ds Cluster Documentation](https://gitlab.fhnw.ch/i4ds/itinfrastructure/howto/-/wikis/services/slurm) 