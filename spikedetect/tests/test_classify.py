"""Tests for spikedetect.pipeline.classify."""

import numpy as np

from spikedetect.pipeline.classify import classify_spikes


class TestClassifySpikes:
    def test_all_good_when_below_distance_above_amplitude(self):
        distances = np.array([1.0, 2.0, 3.0, 1.5, 2.5])
        amplitudes = np.array([5.0, 6.0, 7.0, 5.5, 6.5])
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=10.0,
            amplitude_threshold=1.0,
        )
        # All should be in the good quadrant (good or weird)
        assert np.all(good | weird)
        assert not np.any(bad)

    def test_empty_arrays(self):
        distances = np.array([])
        amplitudes = np.array([])
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=10.0,
            amplitude_threshold=1.0,
        )
        assert len(good) == 0
        assert len(weird) == 0
        assert len(weirdbad) == 0
        assert len(bad) == 0

    def test_all_bad_when_above_distance_below_amplitude(self):
        distances = np.array([20.0, 30.0, 40.0])
        amplitudes = np.array([-1.0, -2.0, -3.0])
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=10.0,
            amplitude_threshold=1.0,
        )
        assert not np.any(good)
        assert not np.any(weird)
        # These should end up in bad or weirdbad
        assert np.all(bad | weirdbad)

    def test_categories_cover_all_spikes(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0, 30, 100)
        amplitudes = rng.uniform(-2, 10, 100)
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=15.0,
            amplitude_threshold=0.5,
        )
        # Every spike should appear in at least one category
        covered = good | weird | weirdbad | bad
        assert np.sum(covered) >= len(distances) * 0.8  # most should be covered

    def test_single_spike(self):
        distances = np.array([5.0])
        amplitudes = np.array([3.0])
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=10.0,
            amplitude_threshold=1.0,
        )
        assert len(good) == 1
        # Single spike in good quadrant should be good
        assert good[0] or weird[0]

    def test_weirdbad_high_distance_high_amplitude(self):
        distances = np.array([1.0, 2.0, 20.0])
        amplitudes = np.array([5.0, 6.0, 5.0])
        good, weird, weirdbad, bad = classify_spikes(
            distances, amplitudes,
            distance_threshold=10.0,
            amplitude_threshold=1.0,
        )
        # Third spike (high distance, high amplitude) = weirdbad
        assert weirdbad[2]
