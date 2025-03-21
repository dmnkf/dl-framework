from src.data.pytorch.datamodules.base_datamodule import BaseDataModule
from src.data.pytorch.cmr_dataset import CMRDataset


class CMRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_class=CMRDataset, *args, **kwargs)
