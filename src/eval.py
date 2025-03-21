import gc
import logging
from typing import Optional
from pathlib import Path

import hydra
import json
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
    extras,
)
from src.utils.instantiators import instantiate_model

logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


@task_wrapper
def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of metrics dictionaries.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)
    datamodule.setup("test")  # train set is needed for model initialization

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate_model(cfg.model, test_dataset=datamodule.test_dataset)

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }

    logger.info("Starting testing...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    logger.info("Testing finished!")
    metrics_dict = trainer.callback_metrics

    logger.info("Cleaning up...")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Goodbye!")

    return metrics_dict, object_dict


def get_json_metrics_dict(metrics_dict: dict) -> dict:
    def tensor_to_builtin(value):
        if isinstance(value, torch.Tensor):
            return value.tolist()  # or value.item() if scalar
        if isinstance(value, dict):
            return {k: tensor_to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [tensor_to_builtin(v) for v in value]
        return value

    return tensor_to_builtin(metrics_dict)


def save_json_metrics_dict(metrics_dict: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(get_json_metrics_dict(metrics_dict), f, indent=4)
    logger.info(f"Metrics dictionary saved to: {path}")


@hydra.main(config_path=str(CONFIG_ROOT), config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = evaluate(cfg)
    metric_dict_path = Path(cfg.paths.output_dir) / "metrics_dict.json"
    save_json_metrics_dict(metric_dict, metric_dict_path)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
