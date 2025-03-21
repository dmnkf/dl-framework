# Based on the implementation of https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/ECGEncoder.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional, Tuple, Union, List

import timm.models.vision_transformer
import torch
import torch.nn as nn

from src.models.encoder_interface import EncoderInterface
from src.utils.model_weights import PretrainedWeightsMixin


class ECGEncoder(
    PretrainedWeightsMixin,
    EncoderInterface,
    timm.models.vision_transformer.VisionTransformer,
):
    def __init__(
        self,
        img_size: Union[Tuple[int, int, int], List[int]],
        patch_size: Union[Tuple[int, int], List[int]],
        embedding_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: Optional[nn.Module],
        global_pool: str,
        pretrained_weights: Optional[str] = None,
    ):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        self.global_pool = global_pool
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.norm_layer = norm_layer

        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
            )
        if self.global_pool:
            self.fc_norm = norm_layer(embedding_dim)
            del self.norm  # remove the original norm

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def forward_features(self, x, localized=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if localized:
            outcome = x[:, 1:]
        elif self.global_pool == "attention_pool":
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(
                q, k, v
            )  # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

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
        if target_key.startswith("head."):
            return None

        return super()._find_matching_state_dict_key(target_key, available_keys)


# Adapted from: https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py#L161
def vit_patchX(**kwargs):
    """Function to create Vision Transformer conforming to the pre-trained weights by Turgut et. al (2025)"""
    model = ECGEncoder(
        patch_size=(1, 100),  # To match patch_embed.proj.weight: [384, 1, 1, 100]
        img_size=(1, 12, 2500),
        embedding_dim=384,  # To match embedding dimension
        depth=3,  # 3 transformer blocks
        num_heads=6,  # 384/64=6 heads (standard head dim of 64)
        mlp_ratio=4,  # Matches the 1536 dimension in mlp layers
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
