from abc import abstractmethod, ABC
import torch


class EncoderInterface(ABC):
    """Interface for neural network encoders that extract features from input data."""

    @abstractmethod
    def forward_features(
        self, x: torch.Tensor, localized: bool = False
    ) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            x: Input tensor
            localized: Whether to return localized features instead of global features

        Returns:
            Tensor of extracted features
        """
        pass
