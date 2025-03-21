import logging
import os
import warnings

import torch
from typing import Any, Callable, Dict, Tuple, Optional
from importlib.util import find_spec

from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.loggers import WandbLogger

from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from rich.prompt import Prompt

from src.data.raw.registry import RawDatasetRegistry
from src.data.unified import UnifiedDataset

logger = logging.getLogger(__name__)


def get_log_level():
    return logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.INFO


def compute_pos_weight(labels: torch.Tensor, clamp_val: float = 100.0) -> torch.Tensor:
    """
    Given a multi-label 0/1 tensor of shape (N, C) for N samples, C classes,
    returns a 1D tensor of shape (C,) for pos_weight in BCEWithLogitsLoss.

    pos_weight[i] = (N - count[i]) / count[i]
    where count[i] is the number of positive samples for class i.
    """
    num_samples, num_classes = labels.shape

    pos_counts = labels.sum(dim=0)  # shape: (C,)
    neg_counts = num_samples - pos_counts  # shape: (C,)

    pos_weight = torch.zeros(num_classes, dtype=torch.float)
    nonzero_mask = pos_counts > 0
    pos_weight[nonzero_mask] = neg_counts[nonzero_mask] / pos_counts[nonzero_mask]

    pos_weight = torch.log1p(pos_weight)
    return pos_weight


def get_accelerator(accelerator_name: str) -> str:
    if accelerator_name == "gpu" or accelerator_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        logger.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        logger.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        logger.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        logger.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            logger.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            logger.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    logger.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        logger.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!\n"
            "Available metrics: {metric_dict.keys()}"
        )

    metric_value = metric_dict[metric_name].item()
    logger.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def log_files_to_logger(logger_instance: Any, files_dict: Dict[str, Path]) -> None:
    """Log files to a specific logger instance.

    Args:
        logger_instance: The logger instance (e.g., WandbLogger)
        files_dict: Dictionary mapping file names to their paths
    """
    # WandbLogger
    if isinstance(logger_instance, WandbLogger):
        for file_name, file_path in files_dict.items():
            if file_path.exists():
                logger_instance.experiment.save(
                    str(file_path), base_path=str(file_path.parent)
                )
                logger.info(f"Uploaded {file_name} to W&B!")
            else:
                logger.warning(f"{file_name} not found at {file_path}!")


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Log hyperparameters using Lightning loggers.

    Args:
        object_dict: Dictionary containing config, model, and trainer objects.
    """
    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams = {
        "model": cfg["model"],
        "data": cfg["data"],
        "trainer": cfg["trainer"],
        "callbacks": cfg.get("callbacks"),
        "task_name": cfg.get("task_name"),
        "ckpt_path": cfg.get("ckpt_path"),
        "seed": cfg.get("seed"),
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "model/params/non_trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "slurm_env": {key: os.environ[key] for key in os.environ if "SLURM" in key},
    }

    data_root_path = Path(cfg.get("data", {}).get("data_root", ""))
    dataset_key = cfg.get("data", {}).get("dataset_key", "")
    unified_dataset = UnifiedDataset(
        data_root_path, RawDatasetRegistry.get_modality(dataset_key), dataset_key
    )
    files_to_log = {
        "splits.json": unified_dataset.paths["misc"]["splits_file"],
        "dataset_info.json": unified_dataset.paths["misc"]["dataset_info"],
        "partitioning_cfg.json": unified_dataset.paths["misc"]["partitioning_cfg"],
    }

    for log_instance in trainer.loggers:
        log_instance.log_hyperparams(hparams)
        log_files_to_logger(log_instance, files_to_log)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else logger.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        logger.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        logger.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
