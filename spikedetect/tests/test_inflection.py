"""Tests for spikedetect.pipeline.inflection."""

import numpy as np

from spikedetect.pipeline.inflection import (
    estimate_spike_times,
    likely_inflection_point,
)


class TestLikelyInflectionPoint:
    def test_returns_valid_index(self, spike_template):
        fs = 10000
        stw = len(spike_template)
        n_candidates = 10
        rng = np.random.default_rng(42)

        # Create waveforms that look like spikes
        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half
        n_window = len(spike_window)

        waveforms = np.zeros((n_window, n_candidates))
        for i in range(n_candidates):
            # Embed the template with slight noise
            waveforms[:, i] = (
                spike_template[:n_window]
                + rng.normal(0, 0.01, n_window)
            )

        dtw_distances = rng.uniform(0.1, 5.0, n_candidates)

        inflection_peak, deriv_2nd = likely_inflection_point(
            waveforms, dtw_distances, stw, fs
        )

        assert isinstance(inflection_peak, int)
        assert inflection_peak >= 0
        assert inflection_peak < n_window
        assert len(deriv_2nd) == n_window

    def test_returns_fallback_for_all_zero_distances(self):
        stw = 51
        fs = 10000
        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half
        n_window = len(spike_window)

        waveforms = np.zeros((n_window, 5))
        dtw_distances = np.zeros(5)

        inflection_peak, _ = likely_inflection_point(
            waveforms, dtw_distances, stw, fs,
        )
        # Should return idx_m fallback
        assert inflection_peak == round(stw * 4 / 5)


class TestEstimateSpikeTimes:
    def test_returns_correct_length(self, spike_template):
        fs = 10000
        stw = len(spike_template)
        n_spikes = 5
        rng = np.random.default_rng(42)

        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half
        n_window = len(spike_window)

        spike_locs = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        waveforms = np.zeros((n_window, n_spikes))
        for i in range(n_spikes):
            waveforms[:, i] = (
                spike_template[:n_window]
                + rng.normal(0, 0.01, n_window)
            )

        dtw_distances = rng.uniform(0.1, 5.0, n_spikes)

        corrected, uncorrected, infl_peak = estimate_spike_times(
            spike_locs, waveforms, dtw_distances, stw, fs,
            distance_threshold=15.0,
        )

        assert len(corrected) == n_spikes
        assert len(uncorrected) == n_spikes
        np.testing.assert_array_equal(uncorrected, spike_locs)
        assert isinstance(infl_peak, int)
