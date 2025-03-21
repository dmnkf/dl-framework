# Based on the implementation of: https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/ResnetEmbeddingModel.py

from typing import Optional
import torch
import torchvision
from torch import nn

from src.models.encoder_interface import EncoderInterface
from src.utils.model_weights import PretrainedWeightsMixin


class CMREncoder(PretrainedWeightsMixin, EncoderInterface, nn.Module):
    BACKBONE_MODELS = ["resnet18", "resnet50"]

    def __init__(
        self, backbone_model_name: str, pretrained_weights: Optional[str] = None
    ):
        super().__init__()
        if backbone_model_name not in self.BACKBONE_MODELS:
            raise ValueError(f"Unknown backbone model: {backbone_model_name}")

        if backbone_model_name == "resnet18":
            resnet = torchvision.models.resnet18()
            self.pooled_dim = 512
        elif backbone_model_name == "resnet50":
            resnet = torchvision.models.resnet50()
            self.pooled_dim = 2048
        else:
            raise ValueError(f"Unknown model type: {backbone_model_name}")

        self.encoder = self._remove_last_layer(resnet)

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def _find_matching_state_dict_key(
        self, target_key: str, available_keys: set
    ) -> Optional[str]:
        """Find matching key by handling the encoder prefix in model's state dict.

        Args:
            target_key: Key from model's state dict (with 'encoder.' prefix)
            available_keys: Keys available in the loaded weights

        Returns:
            Optional[str]: Matching key from available_keys if found, None otherwise
        """
        for prefix in ("encoder.", ""):
            imaging_key = f"encoder_imaging.{target_key[len(prefix):]}"
            if imaging_key in available_keys:
                return imaging_key

        return super()._find_matching_state_dict_key(target_key, available_keys)

    def _remove_last_layer(self, resnet):
        """
        Remove the fully connected layer and pooling layer from the resnet model.
        """
        return nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
