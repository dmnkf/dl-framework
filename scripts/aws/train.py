import json
import logging
from datetime import datetime
from pathlib import Path

import boto3
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import os
import dotenv
import rootutils
from torch._dynamo.utils import print_time_report

dotenv.load_dotenv()

sm_client = boto3.client("sagemaker", region_name=os.getenv("AWS_REGION"))
sts_client = boto3.client("sts", region_name=os.getenv("AWS_REGION"))
session = boto3.session.Session(region_name=os.getenv("AWS_REGION"))
# Fix float casting issue
np.float = float

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)
CONFIG_ROOT = PROJECT_ROOT / "configs"


def load_config_yaml_from_module(module_name, config="default.yaml"):
    """Load config file from a module in the ioptimize.config directory."""
    config_path = Path(module_name.__file__).parent / config
    return OmegaConf.load(config_path)


def save_json_to_s3(data: dict, bucket_name: str, s3_key: str):
    """
    Save a JSON file to an S3 bucket.

    Args:
        data (dict): The data to be saved as JSON.
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The file key in the S3 bucket.

    Returns:
        bool: True if file was uploaded successfully, False otherwise.
    """
    # Initialize a boto3 S3 client
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    # Convert the data to a JSON string
    json_data = json.dumps(data)

    # Upload the JSON string to S3
    s3.put_object(Body=json_data, Bucket=bucket_name, Key=s3_key)


def load_sm_config_if_exists(cfg: DictConfig) -> DictConfig:
    """
    Load the SageMaker configuration from a JSON file if it exists.

    This function attempts to read a JSON configuration file located at
    '/opt/ml/input/data/config/config.json'. If the file exists, it is opened,
    read, and the JSON content is parsed into a Python dictionary. If the file
    does not exist, we return the default config.

    Note:
    This exact functionality is needed to check if configurations were passed
    in from Sagemaker, or if the default configurations should be used.

    Args:
        cfg (DictConfig): Hydra default config.

    Returns:
        dict or None: A dictionary containing the parsed JSON data if the file
        exists and is valid JSON or the default configuration.
    """
    sm_config_path = Path("/opt/ml/input/data/config/config.json")
    if Path(sm_config_path).exists():
        logging.info("Using config provided through SageMaker channel.")
        with open(sm_config_path, "r") as file:
            return OmegaConf.create(json.load(file))
    else:
        logging.info("Using config provided to entrypoint.")
        return cfg


def create_ecr_image_uri(repo_name: str, image_tag: str) -> str:
    """Create ecr image uri from repo_name and image_tag"""
    account = sts_client.get_caller_identity()["Account"]
    region = session.region_name

    # Create ECR image URI
    ecr_image = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{image_tag}"

    return ecr_image


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg):
    """Launch optimization on SageMaker training job."""
    logging.info(OmegaConf.to_yaml(cfg))

    # SageMaker configs
    sm_paths_cfg = cfg.paths

    sm_cfg = cfg.runner
    repo_name = sm_cfg.repo_name
    image_tag = sm_cfg.image_tag
    role_arn = sm_cfg.role_arn
    instance_type = sm_cfg.instance_type
    instance_count = sm_cfg.instance_count
    volume_size_in_gb = sm_cfg.volume_size_in_gb
    max_runtime_in_seconds = sm_cfg.max_runtime_in_seconds

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    training_job_name = f"training-{current_time}"
    # set SM algorithm parameters
    algorithm_image_uri = create_ecr_image_uri(repo_name, image_tag)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    s3_config_key = f"{training_job_name}/config.json"
    logging.info(
        f"Saving config file to S3 bucket {sm_paths_cfg.bucket_name}, key {s3_config_key}."
    )
    save_json_to_s3(cfg, sm_paths_cfg.bucket_name, f"configs/{s3_config_key}")

    # assemble training params for SM training job
    training_params = {
        "AlgorithmSpecification": {
            "TrainingImage": algorithm_image_uri,
            "TrainingInputMode": "File",
            # "ContainerEntrypoint": ["python3"],
            # "ContainerArguments": ["sm_entrypoint.py"],
        },
        "RoleArn": role_arn,
        "ResourceConfig": {
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": volume_size_in_gb,
        },
        "TrainingJobName": training_job_name,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": max_runtime_in_seconds,
        },
        "HyperParameters": {},  # Not using hyperparams as they are supplied through config.
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": sm_paths_cfg.processed_data_dir,
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            },
            {
                "ChannelName": "config",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"{sm_paths_cfg.config_dir}/{s3_config_key}",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": sm_paths_cfg.output_dir,
        },
    }

    logging.info(f"Submitting training job with params: {training_params}")
    sm_client.create_training_job(**training_params)


if __name__ == "__main__":
    main()
