import logging
import sys

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from src.train import main as train_main

logging.basicConfig(level=logging.INFO)


def translate_args(args):
    """Convert SageMaker style arguments (--arg_name arg_value) to Hydra style arguments (arg_name=arg_value).

    Why is this needed?
    Arguments supplied to SageMaker training jobs are passed to the the entrypoint (e.g. `train.py`) in the format
    `train.py --arg1 arg1 --arg2 arg2 ...`. However, an entrypoint configured with Hydra does only accept them as
    `train.py arg1=arg1 arg2=arg2 ...`. This function makes the translation from SageMaker training jobs to
    Hydra entrypoints.
    """
    translated_args = []
    for i in range(0, len(args), 2):
        arg = args[i]
        if arg.startswith("--"):
            arg = arg[2:]  # Remove the '--'
            value = args[i + 1]
            translated_args.append(f"{arg}={value}")
    return translated_args


if __name__ == "__main__":
    # ------- Config Override ---------
    # 1. Take arguments passed through Hyperparameters
    args = translate_args(sys.argv[1:])

    # global initialization
    initialize_config_module(
        version_base=None, config_module="configs", job_name="sagemaker"
    )
    cfg = compose(config_name="config", overrides=args)

    logging.info(f"Configuration file: {OmegaConf.to_yaml(cfg)}")
    # ------- Config Override Done ---------

    train_main(cfg)
