from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_class: type,
        data_root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        use_train_subsample: bool = False,
        cross_validation: bool = False,
        fold_number: Optional[int] = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_train_subsample = use_train_subsample
        self.cross_validation = cross_validation
        self.fold_number = fold_number
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        if self.cross_validation and self.fold_number is None:
            raise ValueError(
                "fold_number must be specified when cross_validation is True"
            )

    def setup(self, stage: Optional[str] = None):
        train_split = (
            f"fold_{self.fold_number}_train" if self.cross_validation else "train"
        )
        val_split = f"fold_{self.fold_number}_val" if self.cross_validation else "val"

        if stage == "evaluation":
            if self.train_dataset is None:
                self.train_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split=train_split,
                    apply_transforms=False,
                    use_train_subsample=self.use_train_subsample,
                    **self.dataset_kwargs,
                )
            if self.val_dataset is None:
                self.val_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split=val_split,
                    apply_transforms=False,
                    **self.dataset_kwargs,
                )
            if not self.cross_validation and self.test_dataset is None:
                self.test_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split="test",
                    apply_transforms=False,
                    **self.dataset_kwargs,
                )

        elif stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split=train_split,
                    use_train_subsample=self.use_train_subsample,
                    **self.dataset_kwargs,
                )
            if self.val_dataset is None:
                self.val_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split=val_split,
                    **self.dataset_kwargs,
                )

        elif stage == "test":
            if not self.cross_validation and self.test_dataset is None:
                self.test_dataset = self.dataset_class(
                    data_root=self.data_root, split="test", **self.dataset_kwargs
                )
            elif self.cross_validation and self.val_dataset is None:
                # In cross-validation mode, use validation set for testing
                self.val_dataset = self.dataset_class(
                    data_root=self.data_root,
                    split=val_split,
                    **self.dataset_kwargs,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.cross_validation:
            return self.val_dataloader()  # Use validation set for testing in CV mode
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
