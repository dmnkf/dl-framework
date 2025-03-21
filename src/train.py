import gc
import logging
from typing import Optional

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


# Fix float casting issue
np.float = float

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)
CONFIG_ROOT = PROJECT_ROOT / "configs"

from src.utils.utils import (
    get_log_level,
    get_metric_value,
    task_wrapper,
    log_hyperparameters,
    extras,
)

from src.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_model,
)

logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of metrics dictionaries.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")  # train set is needed for model initialization

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate_model(cfg.model, train_dataset=datamodule.train_dataset)

    logger.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    log_instances = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate(cfg.trainer, logger=log_instances, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
    }

    if log_instances:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

        for log_instance in log_instances:
            if hasattr(log_instance, "watch") and cfg.model.get("watch_model"):
                logger.info("Watching model with option 'all'")
                log_instance.watch(model, log="all")

    if cfg.get("train"):
        logger.info("Starting training!")
        trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)
        logger.info("Training completed.")
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        logger.info(f"Loading best model from {ckpt_path} for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        logger.info("Testing completed.")
    test_metrics = trainer.callback_metrics

    logger.info("Cleaning up...")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Goodbye!")

    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)

    metric_dict, _ = train(cfg)
    logger.info(f"Final Metrics: {metric_dict}")

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
