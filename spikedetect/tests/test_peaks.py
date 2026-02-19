"""Tests for spikedetect.pipeline.peaks."""

import numpy as np

from spikedetect.pipeline.peaks import find_spike_locations


class TestFindSpikeLocations:
    def test_finds_peaks_in_sinusoid(self):
        fs = 10000
        t = np.arange(fs) / fs  # 1 second
        # Sinusoid with peaks well above mean
        signal = np.sin(2 * np.pi * 50 * t) * 10.0
        locs = find_spike_locations(
            signal, peak_threshold=5.0,
            fs=fs, spike_template_width=51,
        )
        # 50Hz sine should have ~50 peaks in 1 second
        assert len(locs) > 30
        assert len(locs) < 70

    def test_excludes_peaks_near_edges(self):
        fs = 10000
        n = 1000
        signal = np.zeros(n)
        stw = 51
        # Put peaks right at the edges
        signal[10] = 100.0
        signal[n - 10] = 100.0
        # And one in the middle
        signal[500] = 100.0
        locs = find_spike_locations(
            signal, peak_threshold=5.0,
            fs=fs, spike_template_width=stw,
        )
        # Only the middle peak should survive
        assert len(locs) == 1
        assert locs[0] == 500

    def test_returns_empty_for_flat_signal(self):
        signal = np.zeros(10000)
        locs = find_spike_locations(
            signal, peak_threshold=5.0,
            fs=10000, spike_template_width=51,
        )
        assert len(locs) == 0

    def test_returns_int64_dtype(self):
        fs = 10000
        signal = np.zeros(10000)
        signal[5000] = 100.0
        locs = find_spike_locations(
            signal, peak_threshold=5.0,
            fs=fs, spike_template_width=51,
        )
        assert locs.dtype == np.int64

    def test_returns_empty_array_dtype_when_no_peaks(self):
        signal = np.zeros(10000)
        locs = find_spike_locations(
            signal, peak_threshold=5.0,
            fs=10000, spike_template_width=51,
        )
        assert locs.dtype == np.int64
        assert len(locs) == 0

    def test_respects_peak_threshold(self):
        signal = np.zeros(10000)
        signal[3000] = 3.0  # Below threshold
        signal[5000] = 20.0  # Above threshold
        locs = find_spike_locations(
            signal, peak_threshold=10.0,
            fs=10000, spike_template_width=51,
        )
        assert len(locs) == 1
        assert locs[0] == 5000
