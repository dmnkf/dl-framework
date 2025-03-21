import gc
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import hydra
import lightning as pl
import numpy as np
import rootutils
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt

# Fix float casting issue
np.float = float

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)
CONFIG_ROOT = PROJECT_ROOT / "configs"

from src.data.pytorch.base_dataset import SUPPORTED_SPLIT_NAMES, BaseTorchDataset
from src.models.encoder_interface import EncoderInterface
from src.utils.model_weights import PretrainedWeightsMixin
from src.utils.utils import get_log_level, get_accelerator
from src.data.raw.registry import RawDatasetRegistry
from src.data.unified import UnifiedDataset

import src.saliency.gradcam as sali_gradcam
import src.saliency.attention_map as sali_attention_map

logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)


def setup_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Setup and return the datamodule."""
    logger.debug("Initializing datamodule")
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="evaluation")
    logger.info("DataModule setup complete")
    return datamodule


def get_split_dataset(
    datamodule: pl.LightningDataModule, split_name: str
) -> Tuple[BaseTorchDataset, torch.utils.data.DataLoader]:
    """Get dataset and dataloader for a specific split."""
    logger.debug(f"Retrieving dataset for split: {split_name}")
    if split_name == "train":
        return datamodule.train_dataset, datamodule.train_dataloader()
    elif split_name == "val":
        return datamodule.val_dataset, datamodule.val_dataloader()
    elif split_name == "test":
        return datamodule.test_dataset, datamodule.test_dataloader()
    raise ValueError(f"Invalid split name: {split_name}")


def load_model(cfg: DictConfig, device: str) -> torch.nn.Module:
    """Load and return the model from checkpoint."""
    logger.debug(f"Loading model with config: {cfg.model}")

    model = instantiate(cfg.model)
    logger.debug(f"Model instantiated: {type(model)}")

    if not isinstance(model, (PretrainedWeightsMixin, torch.nn.Module)):
        raise ValueError(
            "Model must inherit from PretrainedWeightsMixin and torch.nn.Module"
        )

    if not cfg.ckpt_path:
        raise ValueError("Checkpoint path (ckpt_path) must be provided in config")
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.ckpt_path}")

    logger.debug(f"Loading weights from: {ckpt_path}")
    model.load_pretrained_weights(str(ckpt_path))

    model.eval()
    model.to(device)
    logger.info(f"Model loaded successfully on device: {device}")
    return model


def generate_batch_embeddings(
    model: EncoderInterface, batch: Tuple[torch.Tensor, ...], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate embeddings for a single batch."""
    logger.debug(f"Processing batch of size {batch[0].shape[0]}")
    if len(batch) != 3:
        raise ValueError(
            "Batch must contain input, target, and index in order: [input, target, index]"
        )

    with torch.no_grad():
        inputs = batch[0].to(device)
        indices = batch[-1]
        logger.debug(f"Generating embeddings for indices: {indices.tolist()}")
        batch_embeddings = model.forward_features(inputs)
        return indices, batch_embeddings


def generate_split_embeddings(
    model: EncoderInterface, dataloader: torch.utils.data.DataLoader, device: str
) -> Dict[str, torch.Tensor]:
    """Generate embeddings for an entire dataset split."""
    logger.info(f"Starting embeddings generation for split")
    embeddings = {}
    dataset = dataloader.dataset
    logger.debug(f"Dataset contains {len(dataset)} samples")

    for batch in tqdm(dataloader, desc="Generating embeddings"):
        indices, batch_embeddings = generate_batch_embeddings(model, batch, device)
        for idx, emb in zip(indices, batch_embeddings):
            sample_id = dataset.get_raw_sample_id(idx)
            logger.debug(f"Storing embedding for sample: {sample_id}")
            embeddings[sample_id] = emb.cpu()

    logger.info(f"Generated {len(embeddings)} embeddings for split")
    return embeddings


def generate_gradcam_images(
    model: torch.nn.Module,
    single_sample_batch: torch.Tensor,
    target_labels: List[str],
    dataset: BaseTorchDataset,
    device: str,
) -> Dict[str, np.ndarray]:
    """Generate GradCAM images for a batch."""
    # Workaround for single-sample batches as GradCAM requires at least two samples in a batch
    logger.debug("Duplicating batch for GradCAM workaround")
    single_sample_batch = torch.cat(
        (single_sample_batch, single_sample_batch.to(device)), 0
    )

    gradcam_images = {}
    for target_idx, _ in enumerate(target_labels):
        label = dataset.get_label_by_label_idx(target_idx)
        logger.debug(f"Processing target label: {label} (index {target_idx})")
        try:
            gradcam_image = sali_gradcam.get_gradcam(
                model, single_sample_batch, target_idx
            )[0]
            gradcam_images[label] = gradcam_image
            logger.debug(f"Generated GradCAM for {label}")
        except Exception as e:
            logger.error(f"Failed to generate GradCAM for {label}: {str(e)}")

    return gradcam_images


def generate_attention_map_image(
    model: torch.nn.Module,
    single_sample_batch: torch.Tensor,
    dataset: BaseTorchDataset,
    device: str,
) -> plt.Figure:
    """Generate Attention Map figures for a batch."""
    try:
        figure = sali_attention_map.get_attention_map(
            model, single_sample_batch.to(device)
        )[0]
        return figure
    except Exception as e:
        logger.error(f"Failed to generate Attention Maps: {str(e)}")


def save_attention_map_image(
    figure: plt.Figure, sample_id: str, output_dir: Path
) -> None:
    """Save Attention Map figure to disk."""
    logger.info(f"Saving Attention Map image to: {output_dir}")
    image_path = output_dir / f"{sample_id}_attention.png"
    logger.debug(f"Saving attention map for {sample_id} to: {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    figure.suptitle(f"Attention Map for {sample_id}", fontsize=16)
    figure.savefig(image_path, bbox_inches="tight", dpi=300)
    plt.close(figure)  # Critical for memory management


def save_embeddings(embeddings: Dict, output_path: Path) -> None:
    """Save embeddings to disk."""
    logger.info(f"Saving embeddings to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Embeddings container size: {len(embeddings)}")
    torch.save(embeddings, output_path)
    logger.info(f"Embeddings saved successfully")


def save_gradcam_images(images: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Save GradCAM images to disk."""
    logger.info(f"Saving GradCAM images to: {output_dir}")
    for label, image in images.items():
        image_path = output_dir / f"{label}.png"
        logger.debug(f"Saving image for {label} to: {image_path}")

        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if image.ndim == 2:
            mode = "L"
        elif image.ndim == 3:
            mode = "RGB" if image.shape[2] == 3 else "RGBA"
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode=mode).save(image_path)
    logger.info(f"Saved {len(images)} GradCAM images")


def execute_embedding_pipeline(
    model: EncoderInterface,
    datamodule: pl.LightningDataModule,
    splits: set,
    cfg: DictConfig,
    device: str,
) -> None:
    """End-to-end embedding generation pipeline."""
    logger.info("Starting embedding pipeline")
    try:
        all_embeddings = {}
        for split in splits:
            logger.info(f"Processing split: {split}")
            _, dataloader = get_split_dataset(datamodule, split)
            embeddings = generate_split_embeddings(model, dataloader, device)
            all_embeddings.update(embeddings)

        output_path = (
            Path(cfg.paths.output_dir) / f"{cfg.modality.dataset_key}_embeddings.pt"
        )
        save_embeddings(all_embeddings, output_path)
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("Embedding pipeline completed")


def execute_gradcam_pipeline(
    model: torch.nn.Module,
    datamodule: pl.LightningDataModule,
    cfg: DictConfig,
    record_id: str,
    device: str,
) -> None:
    """End-to-end GradCAM generation pipeline."""
    logger.info(f"Starting GradCAM pipeline for record {record_id}")
    logger.debug("Initializing GradCAM generation")
    if not sali_gradcam.is_supported_model(model):
        logger.warning(f"Model {model.__class__} not supported for GradCAM")
        return None

    try:
        data_root = Path(cfg.data.data_root).absolute()
        unified_dataset = UnifiedDataset(
            data_root,
            RawDatasetRegistry.get_modality(cfg.data.dataset_key),
            cfg.data.dataset_key,
        )

        record_split = unified_dataset.get_split_by_record_id(record_id)
        dataset, _ = get_split_dataset(datamodule, record_split)
        item_idx = dataset.get_idx_by_record_id(record_id)

        logger.debug(f"Retrieved record {record_id} at index {item_idx}")
        single_sample_batch = dataset[item_idx][0].unsqueeze(0).to(device)
        target_labels = unified_dataset.get_dataset_info()["target_labels"]

        gradcam_images = generate_gradcam_images(
            model, single_sample_batch, target_labels, dataset, device
        )
        output_dir = Path(cfg.paths.output_dir).absolute() / "gradcam" / record_id
        save_gradcam_images(gradcam_images, output_dir)
    except Exception as e:
        logger.error(f"GradCAM pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("GradCAM pipeline completed")


def execute_attention_map_pipeline(
    model: torch.nn.Module,
    datamodule: pl.LightningDataModule,
    cfg: DictConfig,
    record_id: str,
    device: str,
) -> None:
    """End-to-end Attention Map generation pipeline."""
    logger.info(f"Starting Attention Map pipeline for record {record_id}")
    logger.debug("Initializing Attention Map generation")
    if not sali_attention_map.is_supported_model(model):
        logger.warning(f"Model {model.__class__} not supported for Attention Maps")
        return {}

    try:
        data_root = Path(cfg.data.data_root).absolute()
        unified_dataset = UnifiedDataset(
            data_root,
            RawDatasetRegistry.get_modality(cfg.data.dataset_key),
            cfg.data.dataset_key,
        )

        record_split = unified_dataset.get_split_by_record_id(record_id)
        dataset, _ = get_split_dataset(datamodule, record_split)
        item_idx = dataset.get_idx_by_record_id(record_id)

        logger.debug(f"Retrieved record {record_id} at index {item_idx}")
        single_sample_batch = dataset[item_idx][0].unsqueeze(0).to(device)

        attention_map = generate_attention_map_image(
            model, single_sample_batch, dataset, device
        )
        output_dir = Path(cfg.paths.output_dir).absolute()
        save_attention_map_image(attention_map, record_id, output_dir)
    except Exception as e:
        logger.error(f"Attention Map pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("Attention Map pipeline completed")


@hydra.main(config_path=str(CONFIG_ROOT), config_name="artifacts", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main execution flow."""
    try:
        logger.debug("Initializing main execution")
        pl.seed_everything(cfg.get("seed", 42))
        device = get_accelerator(cfg.get("accelerator", "auto"))
        logger.debug(f"Using device: {device}")

        datamodule = setup_datamodule(cfg)
        model = load_model(cfg, device)

        if isinstance(model, EncoderInterface) and not cfg.get("record_id"):
            splits = set(cfg.get("splits", SUPPORTED_SPLIT_NAMES))
            execute_embedding_pipeline(model, datamodule, splits, cfg, device)
        elif cfg.get("record_id"):
            execute_gradcam_pipeline(model, datamodule, cfg, cfg.record_id, device)
            execute_attention_map_pipeline(
                model, datamodule, cfg, cfg.record_id, device
            )
        else:
            logger.warning("No valid task specified")

    except Exception as e:
        logger.critical(f"Main execution failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.debug("Cleaning up resources")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
