import numpy as np

from src.data.preprocessing.cmr_preprocessor import normalize_image
from tests.helpers.mock_cmr_data import create_test_image


def test_normalize_image():
    """Test single-slice image normalization for CMR data.

    Goal:
    - Verify that 2D image normalization correctly scales pixel values to [0,1] range
      while preserving relative intensity relationships within a single slice.

    Approach:
    1. Create a realistic test image with tissue-like intensities
    2. Apply normalization to a single slice
    3. Verify output range is exactly [0,1]
    4. Check shape preservation

    This test ensures basic normalization works correctly for the simplest case
    of single-slice processing, which is fundamental for all CMR preprocessing.
    """
    # Test with realistic CMR image
    image = create_test_image(image_size=(50, 50))  # Smaller size for testing
    normalized = normalize_image(image[:, :, 0])  # Test single slice
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalized.shape == (50, 50)


def test_normalize_image_3d():
    """Test normalization behavior with 3D CMR image volumes.

    Goal:
    - Verify that normalization works correctly across multiple slices with
      different intensity distributions.

    Approach:
    1. Create 3D test image with controlled variations between slices
    2. Apply different scaling factors to each slice to test robustness
       - Double values in first slice
       - Add offset to second slice
       - Subtract offset from third slice
    3. Normalize each slice independently
    4. Verify each slice is properly normalized to [0,1]

    This test is crucial because CMR volumes often have intensity variations
    between slices due to different tissue coverage and acquisition factors.
    Each slice must be normalized properly regardless of its original range.
    """
    # Create 3D image with different ranges per slice
    image = create_test_image()

    # Scale slices differently to test normalization
    image[:, :, 0] *= 2.0  # Double the values
    image[:, :, 1] += 0.5  # Add offset
    image[:, :, 2] -= 0.3  # Subtract offset

    # Normalize each slice
    for i in range(image.shape[2]):
        image[:, :, i] = normalize_image(image[:, :, i])

    # Check normalization
    for i in range(3):
        slice_data = image[:, :, i]
        assert np.allclose(slice_data.min(), 0, atol=1e-6)
        assert np.allclose(slice_data.max(), 1, atol=1e-6)
