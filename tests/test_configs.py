import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pytest


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


@pytest.mark.parametrize(
    "model_name",
    [
        "mae",
        "ecg_classifier",
        "ecg_encoder",
        "cmr_classifier",
        "cmr_encoder",
        "sim_clr",
    ],
)
def test_model_train_configs(model_name: str, cfg_train: DictConfig) -> None:
    """Tests different model configurations in training setup.

    :param model_name: Name of the model configuration to test
    :param cfg_train: Base training configuration
    """
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=[f"model={model_name}"],
            return_hydra_config=True,
        )

        assert cfg.model
        assert "_target_" in cfg.model

        HydraConfig().set_config(cfg)

        # Test if model can be instantiated in the training setup
        hydra.utils.instantiate(cfg.data)
        hydra.utils.instantiate(cfg.model)
        hydra.utils.instantiate(cfg.trainer)


@pytest.mark.parametrize(
    "model_name",
    [
        "mae",
        "ecg_classifier",
        "ecg_encoder",
        "cmr_classifier",
        "cmr_encoder",
        "sim_clr",
    ],
)
def test_model_eval_configs(model_name: str, cfg_eval: DictConfig) -> None:
    """Tests different model configurations in evaluation setup.

    :param model_name: Name of the model configuration to test
    :param cfg_eval: Base evaluation configuration
    """
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="eval",
            overrides=[f"model={model_name}", "ckpt_path=."],
            return_hydra_config=True,
        )

        assert cfg.model
        assert "_target_" in cfg.model

        HydraConfig().set_config(cfg)

        # Test if model can be instantiated in the evaluation setup
        hydra.utils.instantiate(cfg.data)
        hydra.utils.instantiate(cfg.model)
        hydra.utils.instantiate(cfg.trainer)
