import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from functools import cached_property, cache
from pathlib import Path
from typing import List

from src.data.raw.ecg.arrhythmia import Arrhythmia, ArrhythmiaLabelMapper
from src.data.raw.registry import RawDatasetRegistry
from src.data.dataset import DatasetModality

logger = logging.getLogger(__name__)


class GroupArrhythmia(Arrhythmia, ABC):
    """Abstract base class for arrhythmia subgroups"""

    def __init__(self, data_root: Path):
        super().__init__(data_root)

    @property
    @abstractmethod
    def dataset_label_group(self) -> str:
        """Set the group name for this subgroup"""
        pass

    @cache
    def _get_group_label_distribution(self):
        """Get the distribution of labels in the subgroup."""
        label_distribution = defaultdict(int)
        for record_id in self.get_all_record_ids():
            for meta in self.get_header(record_id).labels_metadata:
                if meta.get("group").lower() == self.dataset_label_group.lower():
                    label_distribution[
                        meta[ArrhythmiaLabelMapper.INT_CODE_META_KEY]
                    ] += 1
        return dict(label_distribution)

    @property
    def dataset_key(self) -> str:
        """Dynamic dataset key based on group name"""
        return f"arrhythmia_{self.dataset_label_group.lower()}"

    @cached_property
    def paths(self):
        paths = super().paths
        # we keep the paths as they are for the exception of the raw data:
        # the subgroups have the same raw data as the base dataset thus we reuse it
        paths["raw"] = Arrhythmia(self.data_root).paths["raw"]
        return paths

    @cache
    def get_all_record_ids(self) -> List[str]:
        """Filter base records using subgroup criteria"""
        return [
            rid
            for rid in super().get_all_record_ids()
            if any(
                meta.get("group").lower() == self.dataset_label_group.lower()
                for meta in self.get_header(rid).labels_metadata
            )
        ]

    def get_target_labels_by_record(self, record_id: str) -> List[str] | None:
        """Get the target labels for a record, filtered by the subgroup criteria."""
        labels_metadata = self.get_header(record_id).labels_metadata

        # filter out the labels that do not match the subgroup
        group_label_metadata = [
            meta
            for meta in labels_metadata
            if meta.get("group").lower() == self.dataset_label_group.lower()
        ]

        if len(group_label_metadata) == 0:
            return None

        if len(group_label_metadata) > 1:
            # if there are multiple labels, and one of them is the most common label, keep that one
            label_distribution = self._get_group_label_distribution()
            most_common_group_label = max(
                label_distribution, key=label_distribution.get
            )
            logger.debug(
                f"Most common label in group {self.dataset_label_group}: {most_common_group_label}. Distribution: {label_distribution}"
            )

            has_most_common_label = any(
                label_metadata[ArrhythmiaLabelMapper.INT_CODE_META_KEY]
                == most_common_group_label
                for label_metadata in group_label_metadata
            )
            if has_most_common_label:
                logger.debug(
                    f"Record {record_id} has multiple labels in group {self.dataset_label_group}, keeping the most common label: {most_common_group_label}"
                )
                targets = [most_common_group_label]

            else:  # if the most common label is not present, keep the first label alphabetically
                logger.debug(
                    f"Record {record_id} has multiple labels in group {self.dataset_label_group}, keeping the first label alphabetically"
                )
                group_label_metadata.sort(
                    key=lambda x: x[ArrhythmiaLabelMapper.INT_CODE_META_KEY]
                )
                targets = [
                    group_label_metadata[0][ArrhythmiaLabelMapper.INT_CODE_META_KEY]
                ]

        elif len(group_label_metadata) == 1:
            targets = [group_label_metadata[0][ArrhythmiaLabelMapper.INT_CODE_META_KEY]]

        else:  # if no labels are found that can be associated with the subgroup, we have a problem
            raise ValueError(
                f"Record {record_id} has no labels in group {self.dataset_label_group}"
            )

        return targets


@RawDatasetRegistry.register(DatasetModality.ECG.value, "arrhythmia_rhythm")
class RhythmSubgroup(GroupArrhythmia):
    dataset_label_group = "Rhythm"


@RawDatasetRegistry.register(DatasetModality.ECG.value, "arrhythmia_duration")
class DurationSubgroup(GroupArrhythmia):
    dataset_label_group = "Duration"


@RawDatasetRegistry.register(DatasetModality.ECG.value, "arrhythmia_amplitude")
class AmplitudeSubgroup(GroupArrhythmia):
    dataset_label_group = "Amplitude"


@RawDatasetRegistry.register(DatasetModality.ECG.value, "arrhythmia_morphology")
class MorphologySubgroup(GroupArrhythmia):
    dataset_label_group = "Morphology"
