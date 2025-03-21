import numpy as np
import torch
from pytorch_grad_cam import (
    GradCAM,
    KPCA_CAM,
    GradCAMElementWise,
    FullGrad,
    LayerCAM,
    EigenGradCAM,
    EigenCAM,
    XGradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    HiResCAM,
    AblationCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.cmr_classifier import CMRClassifier

SUPPORTED_MODELS = [CMRClassifier]

methods = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "gradcamelementwise": GradCAMElementWise,
    "kpcacam": KPCA_CAM,
}


def is_supported_model(model):
    return model.__class__ in SUPPORTED_MODELS


def get_target_layer(model):
    if model.__class__ == CMRClassifier:
        return model.encoder[7][-1]

    raise ValueError(f"No target layer found for model {model.__class__}")


def get_gradcam(
    model, input_tensor: torch.Tensor, target_idx: int, method: str = "gradcam"
):
    if not is_supported_model(model):
        raise ValueError(
            f"Model {model.__class__} not supported for GradCAM. Supported models: {SUPPORTED_MODELS}"
        )

    target_layers = [get_target_layer(model)]
    targets = [ClassifierOutputTarget(target_idx)]

    batch_visualizations = []
    method = methods[method]
    with method(model=model, target_layers=target_layers) as cam:
        result = cam(input_tensor=input_tensor, targets=targets)
        for i in range(input_tensor.shape[0]):  # iterate over batch
            rgb_img = input_tensor[i].permute(1, 2, 0).cpu().numpy()
            visualization = show_cam_on_image(rgb_img, result[i], use_rgb=True)
            batch_visualizations.append(visualization)
    return np.stack(batch_visualizations)
