from typing import Optional, Tuple, Union

import torch
from torchvision import transforms

from src.data.dataset import DatasetModality
from src.data.pytorch.base_dataset import BaseTorchDataset
from src.utils.ecg_augmentations import CropResizing, FTSurrogate, Jitter, Rescaling


class ECGDataset(BaseTorchDataset):
    """PyTorch Dataset for ECG data."""

    def __init__(
        self,
        data_root: str,
        dataset_key: str,
        input_electrodes: int,
        jitter_sigma: float,
        rescaling_sigma: float,
        time_steps: int,
        ft_surr_phase_noise: float,
        split: Union[str, list],
        device: Optional[Union[str, torch.device]] = "cpu",
        downstream: bool = False,
        apply_transforms: bool = True,
        augmentation_rate: float = 1.0,
        use_train_subsample: bool = False,
    ):
        super().__init__(
            data_root,
            DatasetModality.ECG,
            dataset_key,
            split,
            apply_transforms,
            device,
            use_train_subsample,
        )
        self.input_electrodes = input_electrodes
        self.time_steps = time_steps
        self.downstream = downstream
        self.apply_transforms = apply_transforms
        self.augmentation_rate = augmentation_rate

        # In https://github.com/oetu/mae/blob/ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/util/dataset.py
        #    when training: transform=False & augment=True
        #    when testing: transform=True & augment=False

        self.default_transform = transforms.Compose(
            [
                CropResizing(
                    fixed_crop_len=self.time_steps,
                    start_idx=(
                        None if self.is_training else 0
                    ),  # when transform=True, start_idx=0
                    resize=False,
                )
            ]
        )

        # See configs for explanation of parameters according to paper with quotes.
        self.train_transform = transforms.Compose(
            [
                FTSurrogate(phase_noise_magnitude=ft_surr_phase_noise, prob=0.5),
                Jitter(sigma=jitter_sigma),
                Rescaling(sigma=rescaling_sigma),
            ]
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, targets = super().__getitem__(idx)

        data = data.unsqueeze(
            dim=0
        )  # as first dimension is always the input channels -> 1
        data = data[:, : self.input_electrodes, :]

        data = self.default_transform(data)

        # Note that data augmentation is applied only to the training samples, both during pretraining and fine-tuning. page 5 Turgut et. al (2025)
        if self.is_training and self.apply_transforms:
            data = self.train_transform(data)

        return data, targets, idx
