# Based on
#   https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_pretrain.py
#   https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_pretrain.py

import lightning.pytorch as pl
import timm.optim.optim_factory as optim_factory
import torch
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

from src.models.mae import MaskedAutoencoderViT


class LitMAE(MaskedAutoencoderViT, pl.LightningModule):
    def __init__(
        self,
        img_size,
        patch_size,
        embedding_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_layer,
        norm_pix_loss,
        ncc_weight,
        mask_ratio,
        learning_rate,
        weight_decay,
        warmup_epochs,
        max_epochs,
        pretrained_weights,
        min_lr=0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            ncc_weight=ncc_weight,
            pretrained_weights=pretrained_weights,
        )
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_epochs = max_epochs

        self.mask_ratio = mask_ratio

    def _step(self, batch):
        samples, _, _ = batch
        loss, samples_hat, samples_hat_masked = self(
            samples, mask_ratio=self.mask_ratio
        )

        normalized_corr = self.ncc(samples, samples_hat)
        batch_size = samples.shape[0]
        return loss, normalized_corr, batch_size

    def training_step(self, batch, batch_idx):
        loss, normalized_corr, batch_size = self._step(batch)
        loss_value = loss.item()
        self.log_dict(
            {
                "train/loss": loss_value,
                "train/ncc": normalized_corr,
                "lr": self.optimizers().param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, normalized_corr, batch_size = self._step(batch)
        loss_value = loss.item()
        self.log_dict(
            {
                "val/loss": loss,
                "val/ncc": normalized_corr,
                "lr": self.optimizers().param_groups[0]["lr"],
            },
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss_value

    def configure_optimizers(self):
        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_pretrain.py#L256
        param_groups = optim_factory.add_weight_decay(self, self.weight_decay)

        # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/main_pretrain.py#L257
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.learning_rate, betas=(0.9, 0.95)
        )

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
                "interval": "step",  # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_pretrain.py#L79
                "frequency": 1,
            },
        }

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_pretrain.py#L25
    def norm(self, data: torch.Tensor()) -> torch.Tensor():
        """
        Zero-Normalize data to have mean=0 and standard_deviation=1

        Parameters
        ----------
        data:  tensor
        """
        mean = torch.mean(data, dim=-1, keepdim=True)
        var = torch.var(data, dim=-1, keepdim=True)

        return (data - mean) / (var + 1e-12) ** 0.5

    # https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/engine_pretrain.py#L38
    def ncc(self, data_0: torch.Tensor(), data_1: torch.Tensor()) -> torch.Tensor():
        """
        Zero-Normalized cross-correlation coefficient between two data sets

        Zero-Normalized cross-correlation equals the cosine of the angle between the unit vectors F and T,
        being thus 1 if and only if F equals T multiplied by a positive scalar.

        Parameters
        ----------
        data_0, data_1 :  tensors of same size
        """

        nb_of_signals = 1
        for dim in range(
            data_0.dim() - 1
        ):  # all but the last dimension (which is the actual signal)
            nb_of_signals = nb_of_signals * data_0.shape[dim]

        cross_corrs = (1.0 / (data_0.shape[-1] - 1)) * torch.sum(
            self.norm(data=data_0) * self.norm(data=data_1), dim=-1
        )
        return cross_corrs.sum() / nb_of_signals
