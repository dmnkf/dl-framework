import numpy as np
from typing import Tuple


def create_test_image(
    image_size: Tuple[int, int] = (210, 210),
    n_slices: int = 3,
) -> np.ndarray:
    """Create a test CMR image array with realistic-looking patterns.

    Args:
        image_size: Size of each slice (height, width)
        n_slices: Number of slices

    Returns:
        numpy array of shape (height, width, slices) with values in [0, 1]
    """
    height, width = image_size
    image = np.zeros((height, width, n_slices))

    # Create coordinate grids
    y, x = np.ogrid[-height / 2 : height / 2, -width / 2 : width / 2]
    radius = np.sqrt(x * x + y * y)

    for slice_idx in range(n_slices):
        # Create circular pattern with varying intensity
        circle = np.exp(-((radius - 50 * slice_idx) ** 2) / 1000)

        # Add some texture
        texture = np.random.rand(height, width) * 0.2

        # Combine patterns
        image[:, :, slice_idx] = circle + texture

    # Normalize to [0, 1] range
    image = (image - image.min()) / (image.max() - image.min())

    return image
