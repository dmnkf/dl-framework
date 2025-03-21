# Data Management

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage and version large data files. DVC allows us to version control our data alongside our code while keeping the data files themselves out of Git.

## Tracked Data Locations

Currently, the following data directories are tracked with DVC:

- `data/processed/`  - Contains processed datasets
- `data/interim/`    - Contains preprocessed records for every dataset
- `data/embeddings/` - Contains precomputed embeddings for every dataset in several states and configs
- `data/raw-zips/`   - Contains downloaded raw zip versions of the datasets

## Remote Storage

We use AWS S3 as our remote data store. The data is stored at:
```
s3://fhnw-artifacts/data/dvc/
```

## AWS Authentication

Before using DVC with our S3 remote storage, you need to configure AWS credentials. The easiest way is using the AWS CLI:

1. Install the AWS CLI if you haven't already:
   ```bash
   pip install awscli
   ```

2. Configure your AWS credentials:
   ```bash
   aws configure
   ```
   You will be prompted for:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region
   - Default output format (press Enter for None)

   Contact your project administrator if you need AWS credentials.

3. Verify your configuration:
   ```bash
   aws sts get-caller-identity
   ```
   This should show your AWS account information if configured correctly.

## Common DVC Commands

### Pulling Data from Remote

To get the latest version of the data from the remote storage:

```bash
dvc pull
```

This will download all DVC-tracked files that are not present in your local workspace.

### Adding New Data

To start tracking a new folder or file with DVC:

```bash
# For a folder
dvc add data/new_folder/

# For a single file
dvc add data/new_folder/data.csv
```

After running these commands:

1. DVC will create a corresponding `.dvc` file that should be committed to Git
2. The actual data will be stored in DVC's cache
3. Remember to push your changes to the remote storage using `dvc push`

### Remote Storage Configuration

The project is configured to work with two remote storage options:

1. **Hetzner Storage Box (Default)**
   ```bash
   # Currently configured as default remote
   dvc pull  # Will pull from Hetzner by default
   ```

2. **AWS S3**
   ```bash
   # To use AWS S3 storage instead
   dvc pull -r ipole-aws
   ```

Both configurations are already set up in the `.dvc/config` file. The Hetzner Storage Box is set as the default remote, but the AWS S3 bucket is available as an alternative. Access to either storage requires appropriate permissions from the team.

### Best Practices

1. Always pull the latest data before starting work:
   ```bash
   dvc pull
   ```

2. After adding new data:
   ```bash
   dvc add data/new_folder/
   git add data/new_folder.dvc
   git commit -m "Add new dataset"
   dvc push
   ```

3. Managing Local Cache

   Over time, your local DVC cache may accumulate unused data. Use `dvc gc` (garbage collection) to clean it up:

   ```bash
   # View what would be removed without actually deleting
   dvc gc --workspace --dry

   # Remove files only referenced in workspace
   dvc gc --workspace

   # Keep files from all branches and tags
   dvc gc -aT

   # Also clean remote storage (be careful!)
   dvc gc --workspace --cloud
   ```

   Important options for `dvc gc`:
   
   - `-w`, `--workspace` - keep only files referenced in current workspace
   - `-a`, `--all-branches` - keep files referenced in all Git branches
   - `-T`, `--all-tags` - keep files referenced in all Git tags
   - `-c`, `--cloud` - also remove files from remote storage (use with caution!)
   - `--dry` - show what would be removed without actually deleting
   - `-f`, `--force` - skip confirmation prompt

   > ⚠️ Warning: Using `--cloud` will permanently delete data from remote storage. Make sure you have backups if needed.

### Troubleshooting

If you encounter issues:

1. Ensure you have proper AWS credentials configured
2. Check if the remote storage is correctly configured:
   ```bash
   dvc remote list
   ```
3. Verify that all `.dvc` files are tracked in Git 