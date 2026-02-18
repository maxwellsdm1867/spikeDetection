"""Tests for spikedetect.pipeline.dtw."""

import numpy as np

from spikedetect.pipeline.dtw import dtw_warping_distance


class TestDtwWarpingDistance:
    def test_identical_signals_distance_zero(self):
        signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        dist, warped_r, warped_t = dtw_warping_distance(signal, signal)
        assert dist == 0.0

    def test_squared_euclidean_cost(self):
        # For two single-element signals, distance = (r - t)^2
        r = np.array([3.0])
        t = np.array([1.0])
        dist, _, _ = dtw_warping_distance(r, t)
        assert dist == pytest.approx((3.0 - 1.0) ** 2)

    def test_two_element_hand_calculation(self):
        # r = [0, 1], t = [0, 1]
        # D[0,0] = 0, D[0,1] = 0+1=1, D[1,0] = 0+1=1, D[1,1] = 0+min(1,0,1)=0
        r = np.array([0.0, 1.0])
        t = np.array([0.0, 1.0])
        dist, _, _ = dtw_warping_distance(r, t)
        assert dist == 0.0

    def test_different_signals_positive_distance(self):
        r = np.array([0.0, 1.0, 0.0])
        t = np.array([1.0, 0.0, 1.0])
        dist, _, _ = dtw_warping_distance(r, t)
        assert dist > 0.0

    def test_returns_warped_signals(self):
        r = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, 2.0, 3.0])
        dist, warped_r, warped_t = dtw_warping_distance(r, t)
        assert len(warped_r) == len(warped_t)
        assert len(warped_r) >= max(len(r), len(t))

    def test_symmetric_for_same_length(self):
        r = np.array([1.0, 3.0, 2.0, 5.0])
        t = np.array([2.0, 1.0, 4.0, 3.0])
        dist_rt, _, _ = dtw_warping_distance(r, t)
        dist_tr, _, _ = dtw_warping_distance(t, r)
        # DTW with squared Euclidean is symmetric for same-length signals
        assert dist_rt == pytest.approx(dist_tr)

    def test_different_length_signals(self):
        r = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist, warped_r, warped_t = dtw_warping_distance(r, t)
        assert dist >= 0.0
        assert len(warped_r) == len(warped_t)

    def test_warped_signals_contain_original_values(self):
        r = np.array([1.0, 5.0, 3.0])
        t = np.array([2.0, 4.0, 6.0])
        _, warped_r, warped_t = dtw_warping_distance(r, t)
        # Warped signals should only contain values from the original signals
        assert set(warped_r).issubset(set(r))
        assert set(warped_t).issubset(set(t))


# Import pytest for approx
import pytest
