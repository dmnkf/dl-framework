from typing import List, Optional, Union, Tuple

import torch
import random
import torchvision
from torch import Tensor
from torchvision.transforms import transforms

from src.data.dataset import DatasetModality
from src.data.pytorch.base_dataset import BaseTorchDataset


def to_float_tensor(x):
    return x.float()


class CMRDataset(BaseTorchDataset):
    """PyTorch Dataset for CMR (Cardiac Magnetic Resonance) data."""

    def __init__(
        self,
        data_root: str,
        dataset_key: str,
        augmentation_rate: float,
        img_size: int,
        live_loading: bool,
        split: Union[str, List[str]],
        rotation_degrees: int,
        brightness: float,
        contrast: float,
        saturation: float,
        random_crop_scale: List[float],
        manual_crop: Optional[dict],
        downstream: bool = False,
        apply_transforms: bool = True,
        device: Optional[Union[str, torch.device]] = "cpu",
        use_train_subsample: bool = False,
    ):
        super().__init__(
            data_root,
            DatasetModality.CMR,
            dataset_key,
            split,
            apply_transforms,
            device,
            use_train_subsample,
        )

        self.live_loading = live_loading
        self.img_size = img_size
        self.augmentation_rate = augmentation_rate
        self.downstream = downstream
        self.apply_transforms = apply_transforms
        self.manual_crop = manual_crop

        # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/ContrastiveImageDataset.py#L31
        # ALSO applicable for downstream:
        #   - https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/EvalImageDataset.py#L24 (Downstream)
        #   - https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/utils/utils.py#L70
        self.default_transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.Lambda(to_float_tensor),
            ]
        )

        # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/utils/utils.py#L33
        # ALSO applicable for downstream:
        #   - https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/EvalImageDataset.py#L24 (Downstream)
        #   - https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/utils/utils.py#L70
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_degrees),
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                ),
                transforms.RandomResizedCrop(
                    size=img_size, scale=tuple(random_crop_scale)
                ),
                transforms.Lambda(to_float_tensor),
            ]
        )

    def apply_manual_cropping(self, data: torch.Tensor) -> torch.Tensor:
        # Note: that the "manual cropping" is only happening in the ContrastiveImagingAndECGDataset.py in the MMCL-ECG-CMR repository
        # Manual cropping" of the image:
        # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/ContrastiveImagingAndECGDataset.py#L62
        return torchvision.transforms.functional.crop(
            data,
            top=int(self.manual_crop["top"] * self.img_size),
            left=int(self.manual_crop["left"] * self.img_size),
            height=int(self.manual_crop["height"] * self.img_size),
            width=int(self.manual_crop["width"] * self.img_size),
        )

    def apply_augmentations(
        self, data: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.downstream:
            # see https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/EvalImageDataset.py#L46C5-L49C34
            if random.random() < self.augmentation_rate and self.is_training:
                view = self.transform(data)
            else:
                view = self.default_transform(data)
            return view, view

        else:  # For contrastive learning
            # https://github.com/oetu/MMCL-ECG-CMR/blob/bd3c18672de8e5fa73bb753613df94547bd6245b/mmcl/datasets/ContrastiveImageDataset.py#L58
            view_1 = self.transform(data)
            if random.random() < self.augmentation_rate:
                view_2 = self.transform(data)
            else:
                view_2 = self.default_transform(data)
            return view_1, view_2

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, int] | tuple[Tensor, Tensor, Tensor, int]:
        data, targets = super().__getitem__(idx)

        if self.manual_crop is not None:
            data = self.apply_manual_cropping(data)

        if self.live_loading:
            data = data / 255.0

        if self.apply_transforms:
            view_1, view_2 = self.apply_augmentations(data, targets)
        else:
            data = self.default_transform(data)
            return data, targets, idx

        return view_1, view_2, targets, idx
