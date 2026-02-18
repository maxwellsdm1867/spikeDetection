"""Spike classification into quality categories.

Ports the MATLAB function ``thegoodthebadandtheweird.m`` which uses DTW
distance and amplitude thresholds (with quantile-based refinement) to sort
spike candidates into four categories.
"""

from __future__ import annotations

import numpy as np


def _safe_quantile(arr: np.ndarray, q: float) -> float:
    """Return quantile *q* of *arr*, or NaN if *arr* is empty."""
    if len(arr) == 0:
        return np.nan
    return float(np.quantile(arr, q))


class SpikeClassifier:
    """Quantile-based spike quality classification.

    Ports MATLAB ``thegoodthebadandtheweird.m``. Categorizes spike
    candidates into four quality tiers based on DTW distance and amplitude.

    Examples
    --------
    >>> good, weird, weirdbad, bad = SpikeClassifier.classify(
    ...     distances, amplitudes, dist_thresh=10.0, amp_thresh=0.2)
    """

    @staticmethod
    def classify(
        distances: np.ndarray,
        amplitudes: np.ndarray,
        distance_threshold: float,
        amplitude_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Classify spike candidates into good/weird/weirdbad/bad.

        Parameters
        ----------
        distances : np.ndarray
            1-D DTW distances for each candidate.
        amplitudes : np.ndarray
            1-D amplitudes for each candidate (same length).
        distance_threshold : float
            Maximum DTW distance for "close" match.
        amplitude_threshold : float
            Minimum amplitude for "large enough".

        Returns
        -------
        good : np.ndarray
            Boolean mask -- high-confidence spikes.
        weird : np.ndarray
            Boolean mask -- borderline spikes near cluster edges.
        weirdbad : np.ndarray
            Boolean mask -- ambiguous region candidates.
        bad : np.ndarray
            Boolean mask -- candidates that clearly fail.

        Notes
        -----
        Original MATLAB function: ``thegoodthebadandtheweird.m``
        """
        x = np.asarray(distances, dtype=float)
        y = np.asarray(amplitudes, dtype=float)
        n = len(x)

        if n == 0:
            empty = np.array([], dtype=bool)
            return empty.copy(), empty.copy(), empty.copy(), empty.copy()

        xt = float(distance_threshold)
        yt = float(amplitude_threshold)

        good_quad = (x < xt) & (y > yt)

        q85_x = _safe_quantile(x[good_quad], 0.85)
        q20_y = _safe_quantile(y[good_quad], 0.2)
        weird = good_quad & ((x > q85_x) | (y < q20_y))

        good = (x < xt) & (y > yt) & ~weird
        if np.sum(good) >= 40:
            q20_x_close = _safe_quantile(x[x < xt], 0.2)
            q35_y_high = _safe_quantile(y[y > yt], 0.35)
            good = (x < q20_x_close) & (y > q35_y_high)

        weirdbad = ((y > yt) & (x > xt)) | ((y <= yt) & (y > 0))
        if np.sum(weirdbad) >= 7:
            q85_x_close = _safe_quantile(x[x < xt], 0.85)
            weirdbad = ((y > yt) & (x > xt) & (x < 2 * q85_x_close)) | (
                (y <= yt) & (y > 0)
            )

        bad = ((x > xt) | (y < yt)) & ~weirdbad
        if len(bad) >= 40:
            bad = (x > xt) & (y < yt)

        return good, weird, weirdbad, bad


# Backwards-compatible alias
classify_spikes = SpikeClassifier.classify
