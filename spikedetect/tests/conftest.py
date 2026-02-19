"""Shared pytest fixtures for spikedetect tests."""

import numpy as np
import pytest

from spikedetect.models import Recording, SpikeDetectionParams


@pytest.fixture
def spike_template():
    """Biphasic spike template at 10kHz sampling."""
    fs = 10000
    width = round(0.005 * fs) + 1  # 51 samples
    t = np.linspace(-0.0025, 0.0025, width)
    template = np.exp(-0.5 * (t / 0.0003) ** 2) - 0.3 * np.exp(
        -0.5 * ((t - 0.001) / 0.0005) ** 2
    )
    return template


@pytest.fixture
def default_params(spike_template):
    """Default detection parameters for testing."""
    return SpikeDetectionParams(
        fs=10000,
        spike_template=spike_template.copy(),
        hp_cutoff=200.0,
        lp_cutoff=800.0,
        diff_order=1,
        peak_threshold=5.0,
        distance_threshold=15.0,
        amplitude_threshold=0.2,
        polarity=1,
    )


@pytest.fixture
def synthetic_recording(spike_template):
    """Recording with spikes embedded at known positions.

    Returns a tuple of (Recording, true_spike_positions).
    Embeds 10 spikes into 5 seconds of Gaussian noise at 10kHz.
    """
    fs = 10000
    duration = 5.0
    n_samples = int(fs * duration)
    rng = np.random.default_rng(42)

    # Base noise
    noise = rng.normal(0, 0.01, n_samples)

    # Embed spikes at known positions (well separated, away from edges)
    spike_positions = np.array(
        [5000, 8000, 12000, 18000, 24000,
         30000, 35000, 40000, 43000, 46000]
    )
    half_w = len(spike_template) // 2

    for pos in spike_positions:
        start = pos - half_w
        end = start + len(spike_template)
        # Scale up to be clearly visible
        noise[start:end] += spike_template * 5.0

    recording = Recording(
        name="synthetic_test",
        voltage=noise,
        sample_rate=fs,
    )

    return recording, spike_positions
