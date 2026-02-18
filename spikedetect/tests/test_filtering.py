"""Tests for spikedetect.pipeline.filtering."""

import numpy as np
import pytest

from spikedetect.pipeline.filtering import filter_data


@pytest.fixture
def sample_signal():
    """A 1-second signal with DC offset and some noise at 10kHz."""
    fs = 10000
    t = np.arange(fs) / fs
    rng = np.random.default_rng(123)
    # DC offset + low-freq sine + noise
    signal = 5.0 + 0.5 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 0.01, len(t))
    return signal, fs


class TestFilterData:
    def test_output_length_matches_input(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800)
        assert len(result) == len(signal)

    def test_output_dtype_float64(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800)
        assert result.dtype == np.float64

    def test_hp_filter_removes_dc_offset(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=0)
        # After high-pass filtering, the mean should be near zero (not 5.0)
        # Skip the first 500 samples to avoid transient
        assert abs(np.mean(result[500:])) < 0.1

    def test_polarity_flip(self, sample_signal):
        signal, fs = sample_signal
        pos = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, polarity=1)
        neg = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, polarity=-1)
        np.testing.assert_allclose(pos, -neg)

    def test_diff_order_0_no_differentiation(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=0)
        # First 100 samples should NOT be zeroed for diff_order=0
        assert not np.all(result[:100] == 0)

    def test_diff_order_1_zeros_first_100(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=1)
        np.testing.assert_array_equal(result[:100], 0.0)

    def test_diff_order_2_zeros_first_100(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=2)
        np.testing.assert_array_equal(result[:100], 0.0)

    def test_diff_order_changes_output(self, sample_signal):
        signal, fs = sample_signal
        r0 = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=0)
        r1 = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=1)
        r2 = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=2)
        # They should all be different
        assert not np.allclose(r0[200:], r1[200:])
        assert not np.allclose(r1[200:], r2[200:])

    def test_invalid_diff_order_raises(self, sample_signal):
        signal, fs = sample_signal
        with pytest.raises(ValueError, match="diff_order must be 0, 1, or 2"):
            filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=3)

    def test_1d_output(self, sample_signal):
        signal, fs = sample_signal
        result = filter_data(signal, fs=fs, hp_cutoff=200, lp_cutoff=800)
        assert result.ndim == 1
