"""Peak finding for spike detection.

Ports the MATLAB function ``findSpikeLocations.m``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class PeakFinder:
    """Find candidate spike peaks in filtered electrophysiology data.

    Ports MATLAB ``findSpikeLocations.m``. All methods are static.

    Example::

        >>> locs = PeakFinder.find_spike_locations(
        ...     filtered, peak_threshold=0.001,
        ...     fs=10000, spike_template_width=51,
        ... )
    """

    @staticmethod
    def find_spike_locations(
        filtered_data: np.ndarray,
        peak_threshold: float,
        fs: float,
        spike_template_width: int,
    ) -> np.ndarray:
        """Find candidate spike peak locations in filtered data.

        Args:
            filtered_data: 1-D bandpass-filtered voltage
                data.
            peak_threshold: Height above signal mean that
                a peak must reach.
            fs: Sampling rate in Hz.
            spike_template_width: Half-width of spike
                template. Peaks within this distance of
                signal edges are excluded.

        Returns:
            Sorted 0-based indices (int64) of detected
            peak locations.
        """
        height = np.mean(filtered_data) + peak_threshold
        distance = max(1, round(fs / 1800))

        peak_indices, _ = find_peaks(
            filtered_data, height=height, distance=distance,
        )

        if len(peak_indices) == 0:
            return np.array([], dtype=np.int64)

        n = len(filtered_data)
        mask = (peak_indices >= spike_template_width) & (
            peak_indices < n - spike_template_width
        )
        result = peak_indices[mask].astype(np.int64)
        excluded = len(peak_indices) - len(result)
        if excluded > 0:
            logger.debug(
                "Excluded %d peaks near recording edges (%d kept)",
                excluded, len(result),
            )
        return result


# Backwards-compatible alias
find_spike_locations = PeakFinder.find_spike_locations
