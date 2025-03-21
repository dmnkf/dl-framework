from src.data.pytorch.datamodules.base_datamodule import BaseDataModule
from src.data.pytorch.ecg_dataset import ECGDataset


class ECGDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_class=ECGDataset, *args, **kwargs)
