import logging
from typing import List

from hydra.utils import instantiate
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch

from src.utils.metrics import init_metrics
from src.data.pytorch.base_dataset import BaseTorchDataset

logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config.

    Args:
        callbacks_cfg: DictConfig containing callback configurations.

    Returns:
        List of instantiated callbacks.

    Raises:
        TypeError: If callbacks_cfg is not a DictConfig.
    """
    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return []

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    callbacks = []
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate loggers from config.

    Args:
        logger_cfg: DictConfig containing logger configurations.

    Returns:
        List of instantiated loggers.

    Raises:
        TypeError: If logger_cfg is not a DictConfig.
    """
    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return []

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    loggers = []
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(instantiate(lg_conf))

    return loggers


def instantiate_model(
    model_cfg: DictConfig,
    train_dataset: BaseTorchDataset = None,
    test_dataset: BaseTorchDataset = None,
):
    """Instantiate model from config with preprocessing if needed.

    Args:
        model_cfg: DictConfig containing model configuration
        train_dataset: Optional training dataset for model preprocessing

    Returns:
        Instantiated model
    """
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("Model config must be a DictConfig!")

    cfg = dict(model_cfg)

    # special case for multilabel with positive class weights
    additional_args = {}
    if cfg.get("task_type") == "multilabel":
        logger.info("Preprocessing for multilabel classification model...")
        if train_dataset is not None and hasattr(train_dataset, "y"):
            from src.utils.utils import compute_pos_weight

            pos_weight = compute_pos_weight(train_dataset.y)
            logger.debug(f"Computed pos_weight: {pos_weight}")
        else:
            logger.warning("No train_dataset provided for multilabel task.")
            if hasattr(model_cfg, "num_classes"):
                pos_weight = torch.ones(model_cfg.num_classes)
                logger.warning(
                    f"Setting pos_weight to ones tensor with shape {pos_weight.shape}"
                )
        additional_args["pos_weight"] = pos_weight

    logger.info(f"Instantiating model <{cfg['_target_']}>")
    if len(additional_args) > 0:
        logger.info(f"Additional args for model instantiation: {additional_args}")

    model_instance = instantiate(model_cfg, **additional_args)

    # init metrics
    if not hasattr(model_instance, "num_classes"):
        raise ValueError("Model must have num_classes attribute!")

    if not hasattr(model_instance, "task_type"):
        raise ValueError("Model must have task_type attribute!")

    num_classes = model_instance.num_classes
    task_type = model_instance.task_type

    if train_dataset is not None:
        label_idx_to_class = train_dataset.idx_to_label
        stages = ["train", "val"]
    elif test_dataset is not None:
        label_idx_to_class = test_dataset.idx_to_label
        stages = ["test"]
    else:
        stages = []
        logger.warning(
            "No dataset provided for metrics initialization. No metrics attached to model."
        )

    for stage in stages:
        init_metrics(
            model_instance, stage, num_classes, label_idx_to_class, task=task_type
        )

    return model_instance
