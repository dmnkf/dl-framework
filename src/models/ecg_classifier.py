# Based on the implementation of: https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py

from typing import Dict, List, Optional, Tuple, Union, Literal
import lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from torch import nn

import src.utils.lr_decay as lrd
from src.utils.metrics import MetricsMixin
from src.utils.model_weights import PretrainedWeightsMixin


class ECGClassifier(
    PretrainedWeightsMixin,
    MetricsMixin,
    timm.models.vision_transformer.VisionTransformer,
    pl.LightningModule,
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
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        layer_decay: float,
        norm_layer: nn.Module,
        drop_path_rate: float,
        smoothing: float,
        task_type: Literal["multiclass", "multilabel"],
        global_pool=False,
        masking_blockwise=False,
        mask_ratio=0.0,
        mask_c_ratio=0.0,
        mask_t_ratio=0.0,
        min_lr=0.0,
        pretrained_weights: Optional[str] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        self.save_hyperparameters()

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            drop_rate=drop_path_rate,
            num_classes=num_classes,
        )

        self.pretrained_weights = pretrained_weights
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L373
        self.blocks[-1].attn.forward = ECGClassifier.attention_forward_wrapper(
            self.blocks[-1].attn
        )  # required to read out the attention map of the last layer

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L402
        # manually initialize fc layer (as presumably not part of pretrained weights)
        trunc_normal_(self.head.weight, std=0.01)  # 2e-5)

        self.masking_blockwise = masking_blockwise
        self.mask_ratio = mask_ratio
        self.mask_c_ratio = mask_c_ratio
        self.mask_t_ratio = mask_t_ratio

        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.layer_decay = layer_decay

        self.task_type = task_type
        self.downstream_task = "classification"
        self.pos_weight = pos_weight

        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        if self.global_pool:
            self.fc_norm = norm_layer(embedding_dim)
            del self.norm  # remove the original norm

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L289
        # self.class_weights = 2.0 / (
        #    2.0 * torch.Tensor([1.0, 1.0])
        # )  # total_nb_samples / (nb_classes * samples_per_class)
        self.class_weights = None

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L445
        # We deviate here in favor of BCEWithLogitsLoss which is multi-label compatible.
        if task_type == "multilabel":
            self.criterion = torch.nn.BCEWithLogitsLoss(
                weight=self.class_weights, pos_weight=self.pos_weight
            )
        else:  # multiclass
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=smoothing,
            )

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py#L44
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py#L72
    def random_masking_blockwise(self, x, mask_c_ratio, mask_t_ratio):
        """
        2D: ECG recording (N, 1, C, T) (masking c and t under mask_c_ratio and mask_t_ratio)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        C, T = (
            int(self.img_size[-2] / self.patch_size[-2]),
            int(self.img_size[-1] / self.patch_size[-1]),
        )

        # mask C
        x = x.reshape(N, C, T, D)
        len_keep_C = int(C * (1 - mask_c_ratio))
        noise = torch.rand(N, C, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_C]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        x = torch.gather(x, dim=1, index=index)  # N, len_keep_C(C'), T, D

        # mask T
        x = x.permute(0, 2, 1, 3)  # N C' T D => N T C' D
        len_keep_T = int(T * (1 - mask_t_ratio))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_C, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)  # N T' C' D => N C' T' D

        x_masked = x_masked.reshape(
            N, len_keep_T * len_keep_C, D
        )  # N C' T' D => N L' D

        return x_masked, None, None

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py#L107
    def forward_features(self, x):
        """
        x: [B=N, L, D], sequence
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]
        if self.masking_blockwise:
            x, _, _ = self.random_masking_blockwise(
                x, self.mask_c_ratio, self.mask_t_ratio
            )
        else:
            x, _, _ = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool == "attention_pool":
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

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/models_vit.py#L144
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, :]
            )
        x = self.fc_norm(x)

        if self.downstream_task == "classification":
            return x if pre_logits else self.head(x)
        elif self.downstream_task == "regression":
            return x if pre_logits else self.head(x)  # .sigmoid()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        """Training step for downstream task."""
        x, y, _ = batch

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_finetune.py#L261
        # We inherit forward from VisionTransformer, which calls forward_features and forward_head as per reference above
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        self.compute_metrics(y_hat, y.long(), "train")
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, int], _
    ) -> torch.Tensor:
        """Validation step for downstream task."""
        x, y, _ = batch

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_finetune.py#L261
        # We inherit forward from VisionTransformer, which calls forward_features and forward_head as per reference above
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        self.compute_metrics(y_hat, y.long(), "val")
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, int], _
    ) -> torch.Tensor:
        """Test step for downstream task."""
        x, y, _ = batch

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_finetune.py#L261
        # We inherit forward from VisionTransformer, which calls forward_features and forward_head as per reference above
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        self.compute_metrics(y_hat, y.long(), "test")
        return loss

    def configure_optimizers(self) -> Dict:
        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L438
        param_groups = lrd.param_groups_lrd(
            self,
            self.weight_decay,
            no_weight_decay_list=self.no_weight_decay(),
            layer_decay=self.layer_decay,
        )

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L442C29-L442C33
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/util/lr_sched.py
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.max_epochs,
            eta_min=self.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_finetune.py#L74
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        self.finalize_metrics("train")

    def on_validation_epoch_end(self):
        self.finalize_metrics("val")

    def on_test_epoch_end(self):
        self.finalize_metrics("test")

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_finetune.py#L237C1-L259C22
    @staticmethod
    def attention_forward_wrapper(attn_obj):
        """
        Modified version of def forward() of class Attention() in timm.models.vision_transformer
        """

        def my_forward(x):
            B, N, C = x.shape  # C = embed_dim
            # (3, B, Heads, N, head_dim)
            qkv = (
                attn_obj.qkv(x)
                .reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(
                0
            )  # make torchscript happy (cannot use tensor as tuple)

            # (B, Heads, N, N)
            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
            attn = attn.softmax(dim=-1)
            attn = attn_obj.attn_drop(attn)
            # (B, Heads, N, N)
            attn_obj.attn_map = attn  # this was added

            # (B, N, Heads*head_dim)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x

        return my_forward
