import logging
from typing import Optional, Literal, Dict
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import io

from torchmetrics import Metric
from lightning.pytorch.loggers import NeptuneLogger, WandbLogger

logging.getLogger("matplotlib").setLevel(logging.ERROR)

TaskType = Literal["multiclass", "multilabel"]


def init_metrics(
    module: torch.nn.Module,
    stage: str,
    num_classes: int,
    idx_to_class: dict,
    task: TaskType = "multiclass",
) -> None:
    """Initialize metrics for classification tasks.

    This function sets up appropriate metrics based on the classification task type:
    - For multiclass: Single label per sample, classes are mutually exclusive
    - For multilabel: Multiple labels per sample, classes are independent

    Args:
        module: The PyTorch module to attach metrics to
        stage: Stage name (e.g., 'train' or 'val')
        num_classes: Number of classes/labels
        task: Classification task type ('multiclass' or 'multilabel')

    Example:
        >>> # For multiclass classification (one class per sample)
        >>> init_metrics(model, "train", num_classes=10, task="multiclass")
        >>> # For multilabel classification (multiple classes per sample)
        >>> init_metrics(model, "train", num_classes=10, task="multilabel")
    """
    metric_params = {
        "multiclass": {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro",
        },
        "multilabel": {
            "task": "multilabel",
            "num_labels": num_classes,
            "average": "macro",
        },
    }[task]
    metrics = {}

    # Averaged metrics
    metrics[f"accuracy_{stage}"] = torchmetrics.Accuracy(
        **metric_params,
    )
    metrics[f"precision_{stage}"] = torchmetrics.Precision(**metric_params)
    metrics[f"recall_{stage}"] = torchmetrics.Recall(**metric_params)
    metrics[f"f1_{stage}"] = torchmetrics.F1Score(**metric_params)
    metrics[f"specificity_{stage}"] = torchmetrics.Specificity(**metric_params)

    metrics[f"auroc_{stage}"] = torchmetrics.AUROC(**metric_params)
    metrics[f"average_precision_{stage}"] = torchmetrics.AveragePrecision(
        **metric_params
    )

    # Add per-class metrics (with average=None to get per-class values)
    per_class_params = metric_params.copy()
    per_class_params["average"] = None

    metrics[f"accuracy_class_{stage}"] = torchmetrics.Accuracy(**per_class_params)
    metrics[f"precision_class_{stage}"] = torchmetrics.Precision(**per_class_params)
    metrics[f"recall_class_{stage}"] = torchmetrics.Recall(**per_class_params)
    metrics[f"f1_class_{stage}"] = torchmetrics.F1Score(**per_class_params)
    metrics[f"specificity_class_{stage}"] = torchmetrics.Specificity(**per_class_params)

    metrics[f"auroc_class_{stage}"] = torchmetrics.AUROC(**per_class_params)
    metrics[f"average_precision_class_{stage}"] = torchmetrics.AveragePrecision(
        **per_class_params
    )

    # Attach metrics to module
    setattr(module, f"metrics_{stage}", torch.nn.ModuleDict(metrics))
    setattr(module, "label_idx_to_class", idx_to_class)


class MetricsMixin:
    task_type: str
    label_idx_to_class: dict

    def compute_metrics(
        self, y_hat: torch.Tensor, y_true: torch.Tensor, stage: str
    ) -> None:
        """Compute and update all metrics for a given stage."""
        metrics: Dict[str, Metric] = getattr(self, f"metrics_{stage}")

        for name, metric in metrics.items():
            if isinstance(metric, (torchmetrics.AUROC, torchmetrics.AveragePrecision)):
                # For probabilistic metrics
                if self.task_type == "multiclass":
                    probs = torch.softmax(y_hat, dim=1)
                else:  # multilabel
                    probs = torch.sigmoid(y_hat)
                metric.update(probs, y_true)
            else:
                # For other metrics
                metric.update(y_hat, y_true)

    def finalize_metrics(self, stage: str) -> None:
        """Compute and log metrics with label-aware names."""
        metrics: Dict[str, Metric] = getattr(self, f"metrics_{stage}")

        log_dict = {}
        for key, metric in metrics.items():
            name = key.split(f"_{stage}")[0]
            try:
                metric_value = metric.compute()

                # Handle per-class metrics
                if isinstance(metric_value, torch.Tensor) and metric_value.dim() > 0:
                    for idx, value in enumerate(metric_value):
                        # Get label name or fallback to index
                        label = (
                            self.label_idx_to_class[idx]
                            if self.label_idx_to_class
                            and idx < len(self.label_idx_to_class)
                            else str(idx)
                        )
                        name = name.replace("_class", "")
                        log_key = f"{stage}/{name}/{label}"
                        log_dict[log_key] = value.item()
                else:
                    log_key = f"{stage}/{name}"
                    log_dict[log_key] = (
                        metric_value.item()
                        if isinstance(metric_value, torch.Tensor)
                        else metric_value
                    )

                metric.reset()
            except Exception as e:
                print(f"Error computing metric {name}: {str(e)}")

        self.log_dict(log_dict, prog_bar=True)

    def log_confusion_matrix(
        self,
        logger: Optional[object] = None,
    ) -> Optional[io.BytesIO]:
        """Log confusion matrix at the end of validation epoch.

        For multilabel tasks, this function is a no-op as confusion matrices
        are typically not used (would require one matrix per label).
        """
        if not hasattr(self.metrics_val, "confusion_matrix_val"):
            return None

        conf_matrix_metric = self.metrics_val["confusion_matrix_val"]

        # Skip if no data has been accumulated
        if not conf_matrix_metric._update_count:
            return None

        conf_matrix = conf_matrix_metric.compute()
        conf_matrix_metric.reset()

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
        )
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close()
        buf.seek(0)

        log_key = "val/confusion_matrix"
        if logger is None:
            return buf
        elif isinstance(logger, NeptuneLogger):
            logger.experiment[log_key].upload(buf)
        elif isinstance(logger, WandbLogger):
            logger.log_image(key=log_key, image=[buf])
        return None
