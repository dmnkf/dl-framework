import logging
from collections import OrderedDict
from typing import Dict, Tuple, Any, Optional, Set

import torch

logger = logging.getLogger(__name__)


class PretrainedWeightsMixin:
    """Mixin class for loading pretrained weights with intelligent key matching."""

    def load_pretrained_weights(
        self,
        weights_path: str,
        strict: bool = False,
        missing_key_threshold: float = 0.1,
    ) -> None:
        """Load pretrained weights with intelligent key matching.

        Args:
            weights_path: Path to the pretrained weights file
            strict: Whether to strictly enforce matching keys
            missing_key_threshold: Maximum allowed percentage of missing keys

        Raises:
            RuntimeError: If loading criteria are not met
        """
        try:
            state_dict = self._load_and_extract_state_dict(weights_path)

            target_keys = set(self.state_dict().keys())
            target_keys = self._get_filtered_state_dict_keys(target_keys)

            new_state_dict, stats = self._match_and_validate_state_dict_keys(
                state_dict, target_keys
            )

            self._validate_weights_loading_criteria(
                stats, strict, missing_key_threshold
            )
            self.load_state_dict(new_state_dict, strict=False)

            self._log_weights_loading_results(stats, weights_path)

        except Exception as e:
            logger.error(
                f"Error loading pretrained weights from {weights_path}: {str(e)}"
            )
            raise

    def _load_and_extract_state_dict(self, weights_path: str) -> Dict[str, Any]:
        state_dict = torch.load(weights_path, map_location="cpu")

        # Extract nested state dict if necessary
        for key in ["state_dict", "model", "network"]:
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]

        return state_dict

    def _get_filtered_state_dict_keys(self, target_keys: Set[str]) -> Set[str]:
        before_filter = len(target_keys)
        before_filter_keys = target_keys.copy()
        target_keys = {k for k in target_keys if self._load_state_dict_key(k)}
        if len(target_keys) < before_filter:
            logger.info(
                f"Filtered {before_filter} keys to {len(target_keys)} keys after filtering. "
                f"Filtered keys: {before_filter_keys - target_keys}"
            )
        return target_keys

    def _match_and_validate_state_dict_keys(
        self, source_dict: Dict[str, torch.Tensor], target_keys: Set[str]
    ) -> Tuple[OrderedDict, Dict]:
        new_state_dict = OrderedDict()
        missing_keys = []
        matched_keys = set()
        shape_mismatches = []

        source_keys = set(source_dict.keys())

        for target_key in target_keys:
            matching_key = self._find_matching_state_dict_key(target_key, source_keys)

            if not matching_key:
                missing_keys.append(target_key)
                continue

            if self._shapes_match(
                source_dict[matching_key], self.state_dict()[target_key]
            ):
                new_state_dict[target_key] = source_dict[matching_key]
                matched_keys.add(matching_key)
            else:
                shape_mismatches.append(f"Shape mismatch for {target_key}")

        return new_state_dict, {
            "missing_keys": missing_keys,
            "unexpected_keys": list(source_keys - matched_keys),
            "shape_mismatches": shape_mismatches,
            "total_keys": len(target_keys),
            "matched_keys": len(matched_keys),
        }

    @staticmethod
    def _shapes_match(source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> bool:
        return source_tensor.shape == target_tensor.shape

    def _validate_weights_loading_criteria(
        self, stats: Dict, strict: bool, missing_key_threshold: float
    ) -> None:
        if stats["shape_mismatches"]:
            raise RuntimeError("\n".join(stats["shape_mismatches"]))

        missing_ratio = len(stats["missing_keys"]) / stats["total_keys"]
        if missing_ratio > missing_key_threshold:
            raise RuntimeError(
                f"Too many missing keys: {len(stats['missing_keys'])}/{stats['total_keys']} "
                f"({missing_ratio:.1%} > {missing_key_threshold:.1%}) threshold. "
                f"Missing keys: {stats['missing_keys']}"
            )

        if strict and stats["missing_keys"]:
            raise RuntimeError(f"Strict loading failed: {stats['missing_keys']}")

    def _log_weights_loading_results(self, stats: Dict, weights_path: str) -> None:
        if stats["missing_keys"]:
            logger.warning(
                f"Missing keys: {len(stats['missing_keys'])}/{stats['total_keys']} "
                f"({len(stats['missing_keys']) / stats['total_keys']:.1%})."
                f"Missing keys: {stats['missing_keys']}"
            )
        if stats["unexpected_keys"]:
            logger.warning(f"Unexpected keys: {stats['unexpected_keys']}")

        logger.info(
            f"Successfully loaded weights from {weights_path} "
            f"({stats['matched_keys']}/{stats['total_keys']} layers). "
            f"Coverage: {stats['matched_keys'] / stats['total_keys']:.1%}"
        )

    def _find_matching_state_dict_key(
        self, target_key: str, available_keys: Set[str]
    ) -> Optional[str]:
        """Find matching key by handling the encoder prefix in model's state dict. Can be overridden."""
        if target_key in available_keys:
            return target_key
        return None

    def _load_state_dict_key(self, key: str) -> bool:
        """Filter keys to load. Can be overridden."""
        return True
