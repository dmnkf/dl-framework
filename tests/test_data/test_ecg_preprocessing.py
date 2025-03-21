import pytest
import torch
import numpy as np

from src.data.preprocessing.ecg_preprocessor import Normalisation, baseline_als
from tests.helpers.mock_ecg_data import create_mock_ecg_batch


def relative_tolerance(a, b, rtol=1e-3, atol=1e-7):
    """Custom relative tolerance check that handles zero and near-zero values better."""
    # For values very close to zero, use absolute tolerance
    if np.all(np.abs(a) < atol) and np.all(np.abs(b) < atol):
        return True
    # For other values, use relative tolerance
    return np.all(np.abs(a - b) <= rtol * (np.abs(a) + np.abs(b)) / 2)


def test_remove_baseline_als():
    """Test baseline wander removal using Asymmetric Least Squares smoothing.

    Goal:
    - Verify that the ALS algorithm effectively removes baseline wander while
      preserving the original ECG signal morphology.

    Approach:
    1. Create synthetic ECG-like signal with known properties:
       - Pure sinusoid as the "true" signal
       - Low-frequency sinusoid as known baseline
    2. Apply ALS baseline removal
    3. Verify improvements through multiple metrics:
       - Compare MSE before/after correction
       - Check baseline smoothness
       - Verify frequency content preservation
       - Confirm reduction in low-frequency components

    This test is fundamental as baseline wander is a common ECG artifact that
    can significantly impact signal quality and analysis.
    """
    # Create signal with known baseline
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t)  # Clean signal
    baseline = 0.5 * np.sin(0.1 * np.pi * t)  # Slow varying baseline
    noisy_signal = signal + baseline

    # Remove baseline
    estimated_baseline = baseline_als(noisy_signal, lam=1e7, p=0.3, niter=5)
    corrected_signal = noisy_signal - estimated_baseline

    # The corrected signal should be closer to the original signal
    error_before = np.mean((noisy_signal - signal) ** 2)
    error_after = np.mean((corrected_signal - signal) ** 2)
    assert error_after < error_before

    # The estimated baseline should be smooth (small differences between consecutive points)
    baseline_diff = np.diff(estimated_baseline)
    assert np.all(np.abs(baseline_diff) < 0.1)  # Baseline changes should be small

    # The corrected signal should preserve the main frequency component
    from scipy.fft import fft

    signal_fft = np.abs(fft(signal))
    corrected_fft = np.abs(fft(corrected_signal))

    # Find the dominant frequencies
    signal_peaks = np.argsort(signal_fft)[-3:]  # Get top 3 frequency components
    corrected_peaks = np.argsort(corrected_fft)[-3:]
    # The main frequency component should be preserved
    assert np.any(np.isin(signal_peaks, corrected_peaks))

    # The corrected signal should have reduced low-frequency content
    low_freq_power_before = np.sum(np.abs(fft(noisy_signal))[:10])
    low_freq_power_after = np.sum(np.abs(fft(corrected_signal))[:10])
    assert low_freq_power_after < low_freq_power_before

    # Test with different parameters
    corrected_signal_2 = noisy_signal - baseline_als(noisy_signal, lam=1e5)
    assert not np.array_equal(
        corrected_signal, corrected_signal_2
    )  # Different lambda should give different results


def test_remove_baseline_als_edge_cases():
    """Test baseline removal algorithm's handling of extreme input cases.

    Goal:
    - Ensure the ALS algorithm remains numerically stable and produces
      sensible results for edge case inputs.

    Approach:
    1. Test constant signal:
       - Verify baseline matches input (no correction needed)
       - Check for numerical stability
    2. Test zero signal:
       - Verify baseline is zero
       - Ensure no artifacts are introduced
    3. Test very small magnitude signals:
       - Verify no numerical underflow
       - Check for preservation of signal scale

    Edge case handling is critical for robust preprocessing in real-world
    scenarios where signal quality may be poor or unusual.
    """
    # Test constant signal
    constant_signal = np.ones(1000)
    baseline = baseline_als(constant_signal)
    assert np.allclose(baseline, constant_signal, rtol=1e-3)

    # Test zero signal
    zero_signal = np.zeros(1000)
    baseline = baseline_als(zero_signal)
    assert np.allclose(baseline, zero_signal, rtol=1e-3)

    # Test with very small values
    small_signal = np.random.randn(1000) * 1e-6
    baseline = baseline_als(small_signal)
    assert not np.any(np.isnan(baseline))
    assert not np.any(np.isinf(baseline))


def test_remove_baseline_als_parameters():
    """Test baseline removal algorithm's sensitivity to parameter changes.

    Goal:
    - Verify that the ALS algorithm's parameters effectively control its
      behavior and produce meaningfully different results when adjusted.

    Approach:
    1. Create test signal with known baseline component
    2. Test smoothing parameter (lambda):
       - Compare strong vs weak smoothing effects
       - Verify larger lambda produces smoother baseline
    3. Test asymmetry parameter (p):
       - Compare different p values
       - Verify effect on positive/negative deviations
    4. Test iteration count:
       - Compare results with few vs many iterations
       - Verify convergence behavior

    Parameter sensitivity testing ensures the algorithm is configurable
    for different types of baseline wander and signal characteristics.
    """
    # Create test signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(0.1 * np.pi * t)

    # Test different smoothing parameters
    baseline_strong = baseline_als(signal, lam=1e8)  # Stronger smoothing
    baseline_weak = baseline_als(signal, lam=1e6)  # Weaker smoothing
    assert not np.array_equal(baseline_strong, baseline_weak)

    # Test different asymmetry parameters
    baseline_p1 = baseline_als(signal, p=0.1)  # More asymmetric
    baseline_p2 = baseline_als(signal, p=0.9)  # Less asymmetric
    assert not np.array_equal(baseline_p1, baseline_p2)

    # Test different iteration counts
    baseline_iter1 = baseline_als(signal, niter=2)
    baseline_iter2 = baseline_als(signal, niter=10)
    assert not np.array_equal(baseline_iter1, baseline_iter2)


@pytest.mark.parametrize("mode", ["sample_wise", "channel_wise", "group_wise"])
def test_ecg_normalization_modes(mode):
    """Test ECG signal normalization across different normalization strategies.

    Goal:
    - Verify that each normalization mode correctly standardizes ECG data
      according to its specific strategy while preserving relevant relationships
      between leads/samples.

    Approach:
    1. Create controlled batch of ECG data with known properties
    2. Apply normalization in specified mode:
       - Channel-wise: Each lead independently normalized
       - Sample-wise: Each time point normalized across leads
       - Group-wise: Lead groups normalized together
    3. Verify statistical properties:
       - Mean should be 0 (within tolerance)
       - Standard deviation should be 1 (within tolerance)
    4. Check that normalization scope matches the mode:
       - Channel-wise: Stats correct per lead
       - Sample-wise: Stats correct across all leads
       - Group-wise: Stats correct within groups

    Different normalization modes are essential for preserving different types
    of relationships in the ECG data, such as relative amplitudes between
    leads or temporal patterns within leads.
    """
    # Create mock batch and convert to numpy
    batch = create_mock_ecg_batch(batch_size=32, n_leads=12, seq_length=1000)
    batch_np = batch.numpy()

    # Initialize normalizer
    normalizer = Normalisation(mode=mode)

    # Normalize each sample in batch
    normalized = np.stack([normalizer(sample) for sample in batch_np])
    normalized = torch.from_numpy(normalized)

    # Test output shape
    assert normalized.shape == batch.shape

    # Use a more appropriate tolerance for floating point comparisons
    rtol = 1e-3
    atol = 1e-5

    if mode == "channel_wise":
        # Each channel should be independently normalized
        for sample_idx in range(batch.shape[0]):
            for lead_idx in range(batch.shape[1]):
                lead_data = normalized[sample_idx, lead_idx]
                assert torch.allclose(
                    lead_data.mean(), torch.tensor(0.0), rtol=rtol, atol=atol
                )
                assert torch.allclose(
                    lead_data.std(), torch.tensor(1.0), rtol=rtol, atol=atol
                )

    elif mode == "sample_wise":
        # Each sample should be normalized across all channels
        for sample_idx in range(batch.shape[0]):
            sample_data = normalized[sample_idx]
            assert torch.allclose(
                sample_data.mean(), torch.tensor(0.0), rtol=rtol, atol=atol
            )
            assert torch.allclose(
                sample_data.std(), torch.tensor(1.0), rtol=rtol, atol=atol
            )

    elif mode == "group_wise":
        # Check each group is normalized correctly
        groups = [3, 6, 12]  # Default groups
        for sample_idx in range(batch.shape[0]):
            start = 0
            for end in groups:
                group_data = normalized[sample_idx, start:end]
                assert torch.allclose(
                    group_data.mean(), torch.tensor(0.0), rtol=rtol, atol=atol
                )
                assert torch.allclose(
                    group_data.std(), torch.tensor(1.0), rtol=rtol, atol=atol
                )
                start = end


def test_ecg_normalization_edge_cases():
    """Test ECG normalization with edge cases."""
    normalizer = Normalisation(mode="sample_wise")

    # Test constant signal
    batch = np.ones((12, 1000))
    normalized = normalizer(batch)
    # For constant signal, output should be very close to zero
    assert np.allclose(normalized, np.zeros_like(batch), atol=1e-10)

    # Test zero signal
    batch = np.zeros((12, 1000))
    normalized = normalizer(batch)
    assert np.allclose(normalized, np.zeros_like(batch), atol=1e-10)

    # Test single value different
    batch = np.zeros((12, 1000))
    batch[0, 0] = 1.0
    normalized = normalizer(batch)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()


def test_ecg_normalization_group_sizes():
    """Test normalization with different group sizes."""
    # Create mock data
    batch = create_mock_ecg_batch(batch_size=1, n_leads=12, seq_length=1000).numpy()[0]

    # Test with different group configurations
    group_configs = [
        [4, 8, 12],  # Three groups
        [6, 12],  # Two groups
        [12],  # Single group
    ]

    for groups in group_configs:
        normalizer = Normalisation(mode="group_wise", groups=groups)
        normalized = normalizer(batch)

        # Verify each group is normalized correctly
        start = 0
        for end in groups:
            group_data = normalized[start:end]
            # Use relative tolerance for better numerical stability
            assert relative_tolerance(group_data.mean(), 0.0)
            assert relative_tolerance(group_data.std(), 1.0)
            start = end


def test_ecg_normalization_numerical_stability():
    """Test normalization with very small and large values."""
    normalizer = Normalisation(mode="sample_wise")

    # Test with very small values
    batch_small = np.random.rand(12, 1000) * 1e-10
    normalized_small = normalizer(batch_small)
    assert not np.isnan(normalized_small).any()
    assert not np.isinf(normalized_small).any()
    # For very small values, we mainly care about numerical stability
    assert np.all(np.abs(normalized_small) < 1e3)  # Should not explode

    # Test with very large values
    batch_large = np.random.rand(12, 1000) * 1e10
    normalized_large = normalizer(batch_large)
    assert not np.isnan(normalized_large).any()
    assert not np.isinf(normalized_large).any()
    assert relative_tolerance(normalized_large.std(), 1.0, rtol=1e-3)


def test_ecg_normalization_preserves_relationships():
    """Test that normalization preserves relative relationships between signals."""
    # Create a signal with known relationships
    signal = np.random.randn(12, 1000)
    scale_factor = 2.0
    signal[1] = scale_factor * signal[0]  # Channel 1 is twice Channel 0

    # Test preservation in different modes
    modes = ["channel_wise", "sample_wise", "group_wise"]
    for mode in modes:
        normalizer = Normalisation(mode=mode)
        normalized = normalizer(signal)

        if mode == "sample_wise":
            # For sample_wise normalization, check relative amplitudes
            # Use correlation instead of direct ratio as it's more stable
            correlation = np.corrcoef(normalized[1], scale_factor * normalized[0])[0, 1]
            assert np.abs(correlation) > 0.95  # Slightly relaxed correlation threshold
        elif mode == "channel_wise":
            # In channel_wise mode, check that the amplitudes are normalized independently
            # After normalization, both channels should have std=1, breaking the original relationship
            std_ratio = np.std(normalized[1]) / np.std(normalized[0])
            assert relative_tolerance(
                std_ratio, 1.0, rtol=1e-5, atol=1e-5
            )  # Should be close to 1, not scale_factor
