# Based on the implementation of: https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/SimCLR.py

from typing import List, Tuple, Dict, Optional

import lightning as pl
import torch
import torchmetrics

from src.utils.ntx_ent_loss_custom import NTXentLoss

# from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers import WandbLogger
from neptune.types import File

from src.models.cmr_encoder import CMREncoder
from src.models.linear_classifier import LinearClassifier


class SimCLR(pl.LightningModule):
    """
    Lightning module for imaging SimCLR.

    Alternates training between contrastive model and online classifier.
    """

    def __init__(
        self,
        encoder_backbone_model_name: str,
        projection_dim: int,
        temperature: float,
        num_classes: int,
        init_strat: str,
        weights: Optional[List[float]],
        learning_rate: float,
        weight_decay: float,
        lr_classifier: float,
        weight_decay_classifier: float,
        scheduler: str,
        anneal_max_epochs: int,
        warmup_epochs: int,
        max_epochs: int,
        check_val_every_n_epoch: int,
        log_images: bool = False,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_backbone_model_name = encoder_backbone_model_name
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.num_classes = num_classes
        self.init_strat = init_strat
        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_classifier = lr_classifier
        self.weight_decay_classifier = weight_decay_classifier
        self.scheduler = scheduler
        self.anneal_max_epochs = anneal_max_epochs
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.log_images = log_images

        # Manual optimization for multiple optimizers
        self.automatic_optimization = False

        # Initialize encoder
        self.encoder_imaging = CMREncoder(
            backbone_model_name=encoder_backbone_model_name,
            pretrained_weights=pretrained_weights,
        )
        pooled_dim = self.encoder_imaging.pooled_dim

        self.projection_head = SimCLRProjectionHead(
            pooled_dim, pooled_dim, projection_dim
        )
        self.criterion_train = NTXentLoss(temperature=temperature)
        self.criterion_val = NTXentLoss(temperature=temperature)

        # Defines weights to be used for the classifier in case of imbalanced data
        if not weights:
            weights = [1.0 for _ in range(num_classes)]
        self.weights = torch.tensor(weights)

        # Classifier
        self.classifier = LinearClassifier(
            in_size=pooled_dim, num_classes=num_classes, init_type=init_strat
        )
        self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

        self.top1_acc_train = torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=num_classes
        )
        self.top1_acc_val = torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=num_classes
        )

        self.top5_acc_train = torchmetrics.Accuracy(
            task="multiclass", top_k=5, num_classes=num_classes
        )
        self.top5_acc_val = torchmetrics.Accuracy(
            task="multiclass", top_k=5, num_classes=num_classes
        )

        self.f1_train = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.f1_val = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.classifier_acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.classifier_acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection of data.
        """
        x = self.encoder_imaging(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], _
    ) -> torch.Tensor:
        """
        Alternates calculation of loss for training between contrastive model and online classifier.
        """
        x0, x1, y, indices = batch

        opt1, opt2 = self.optimizers()

        # Train contrastive model using opt1
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        loss, _, _ = self.criterion_train(z0, z1)

        self.log(
            "imaging.train.loss", loss, on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            "imaging.train.top1",
            self.top1_acc_train,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "imaging.train.top5",
            self.top5_acc_train,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # Train classifier using opt2
        embedding = torch.squeeze(self.encoder_imaging(x0))
        y_hat = self.classifier(embedding)
        cls_loss = self.classifier_criterion(y_hat, y)

        y_hat = y_hat.argmax(dim=1)
        y = y.argmax(dim=1)

        self.f1_train(y_hat, y)
        self.classifier_acc_train(y_hat, y)

        self.log(
            "classifier.train.loss",
            cls_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "classifier.train.f1",
            self.f1_train,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "classifier.train.accuracy",
            self.classifier_acc_train,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        opt2.zero_grad()
        self.manual_backward(cls_loss)
        opt2.step()

        return loss

    def validation_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], _
    ) -> torch.Tensor:
        """
        Validate both contrastive model and classifier
        """
        x0, x1, y, indices = batch

        # Validate contrastive model
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss, _, _ = self.criterion_val(z0, z1)

        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("imaging.val.loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(
            "imaging.val.top1",
            self.top1_acc_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "imaging.val.top5",
            self.top5_acc_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # Validate classifier
        self.classifier.eval()
        embedding = torch.squeeze(self.encoder_imaging(x0))
        y_hat = self.classifier(embedding)
        loss = self.classifier_criterion(y_hat, y)

        y_hat = y_hat.argmax(dim=1)
        y = y.argmax(dim=1)

        self.f1_val(y_hat, y)
        self.classifier_acc_val(y_hat, y)

        self.log(
            "classifier.val.loss", loss, on_epoch=True, on_step=False, sync_dist=True
        )
        self.log(
            "classifier.val.f1",
            self.f1_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "classifier.val.accuracy",
            self.classifier_acc_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.classifier.train()

        if not hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(x0)
        return x0

    def on_validation_epoch_end(self) -> None:
        """
        Log an image from each validation step using the appropriate logger.
        """
        if self.log_images and hasattr(self, "validation_step_outputs"):
            example_img = (
                self.validation_step_outputs[0]
                .cpu()
                .detach()
                .numpy()[0][0]  # First image in batch, first channel
            )

            if isinstance(self.logger, NeptuneLogger):
                self.logger.run["Image Example"].upload(File.as_image(example_img))
            elif isinstance(self.logger, WandbLogger):
                self.logger.log_image(key="Image Example", images=[example_img])

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model and online classifier.
        Scheduler for online classifier often disabled
        """
        optimizer = torch.optim.Adam(
            [
                {"params": self.encoder_imaging.parameters()},
                {"params": self.projection_head.parameters()},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr_classifier,
            weight_decay=self.weight_decay_classifier,
        )

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.anneal_max_epochs, eta_min=0, last_epoch=-1
            )
        elif self.scheduler == "anneal":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs
            )

        classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            classifier_optimizer,
            patience=int(20 / self.check_val_every_n_epoch),
            min_lr=self.lr_classifier * 0.0001,
        )

        return (
            {"optimizer": optimizer, "lr_scheduler": scheduler},  # Contrastive
            {"optimizer": classifier_optimizer},  # Classifier
        )
