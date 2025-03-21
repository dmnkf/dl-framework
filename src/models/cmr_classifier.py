# Based on the implementation of: https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/ResnetEvalModel.py

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Literal

import lightning as pl
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from src.models.cmr_encoder import CMREncoder
from src.utils.metrics import MetricsMixin


class CMRClassifier(CMREncoder, MetricsMixin, pl.LightningModule):
    def __init__(
        self,
        backbone_model_name: str,
        num_classes: int,
        weights: Optional[List[float]],
        learning_rate: float,
        weight_decay: float,
        scheduler: str,
        anneal_max_epochs: int,
        warmup_epochs: int,
        max_epochs: int,
        freeze_encoder: bool,
        classifier_type: str,
        task_type: Literal["multiclass", "multilabel"] = "multiclass",
        pretrained_weights: Optional[str] = None,
    ):
        self.task_type = task_type
        if task_type != "multiclass":
            raise ValueError("CMRClassifier only supports multiclass classification")

        super().__init__(backbone_model_name, pretrained_weights)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.anneal_max_epochs = anneal_max_epochs
        self.scheduler = scheduler
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        #  https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/models/ResnetEvalModel.py#L50
        #  NOTE: We specifically omit the possibility of using a projection head as the available model weights come all without it

        # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/models/ResnetEvalModel.py#L77
        input_dim = self.pooled_dim
        if classifier_type == "mlp":
            self.head = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear(input_dim, input_dim // 4)),
                        ("relu1", nn.ReLU(inplace=True)),
                        ("fc2", nn.Linear(input_dim // 4, input_dim // 16)),
                        ("relu2", nn.ReLU(inplace=True)),
                        ("fc3", nn.Linear(input_dim // 16, num_classes)),
                    ]
                )
            )
        else:
            self.head = nn.Linear(input_dim, num_classes)

        # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/models/Evaluator.py#L41
        if weights:
            self.weights = torch.tensor(weights)
        else:
            self.weights = None
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classifier."""
        x = super().forward(x)
        x = self.head(x)
        return x

    def training_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], _
    ) -> torch.Tensor:
        """Training step for classification."""
        x0, _, y, _ = batch
        y_hat = self(x0)
        y_true = y.argmax(dim=1)

        loss = self.criterion(y_hat, y_true)
        self.compute_metrics(y_hat, y_true, "train")
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], _
    ) -> torch.Tensor:
        """Validation step for classification."""
        x0, _, y, _ = batch
        y_hat = self(x0)
        y_true = y.argmax(dim=1)

        loss = self.criterion(y_hat, y_true)
        self.compute_metrics(y_hat, y_true, "val")
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def test_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], _
    ) -> torch.Tensor:
        """Test step for classification."""
        x0, _, y, _ = batch
        y_hat = self(x0)
        y_true = y.argmax(dim=1)

        loss = self.criterion(y_hat, y_true)
        self.compute_metrics(y_hat, y_true, "test")
        self.log("test_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict:
        """Configure optimizer for classification task."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.anneal_max_epochs, eta_min=0, last_epoch=-1
            )

        elif self.scheduler == "anneal":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_epoch_end(self):
        self.finalize_metrics("train")

    def on_validation_epoch_end(self):
        self.finalize_metrics("val")

    def on_test_epoch_end(self):
        self.finalize_metrics("test")
