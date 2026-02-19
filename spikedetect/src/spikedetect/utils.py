"""Waveform smoothing and differentiation utilities.

Ports the MATLAB functions ``smoothAndDifferentiate.m`` and
``Differentiate.m``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


class WaveformProcessor:
    """Waveform smoothing and differentiation operations.

    All methods are static — no instance state is needed. The class
    provides a namespace to avoid function-name collisions.

    Original MATLAB functions: ``smooth``, ``smoothAndDifferentiate.m``,
    ``Differentiate.m``.
    """

    @staticmethod
    def smooth(x: np.ndarray, window_size: int) -> np.ndarray:
        """Moving average smoothing (MATLAB ``smooth`` equivalent).

        Uses ``scipy.ndimage.uniform_filter1d`` with ``mode='nearest'`` to
        match MATLAB's ``smooth`` boundary handling.

        Args:
            x: 1-D input array.
            window_size: Number of samples in the moving-average window.

        Returns:
            Smoothed 1-D float64 array with the same length as *x*.
        """
        return uniform_filter1d(
            np.asarray(x, dtype=np.float64), size=window_size, mode="nearest"
        )

    @staticmethod
    def smooth_and_differentiate(
        waveform: np.ndarray, smooth_window: int
    ) -> np.ndarray:
        """Smooth and compute the 2nd derivative of a spike waveform.

        Applies sequential diff-smooth-diff-smooth operations to extract a
        smoothed second derivative, suitable for inflection-point estimation.
        Direct port of MATLAB ``smoothAndDifferentiate.m``.

        Args:
            waveform: 1-D spike waveform (voltage vs. time).
            smooth_window: Window size for moving-average smoothing steps.

        Returns:
            Smoothed second derivative, same length as *waveform*. The first
            two samples are zero (padding to preserve alignment).
        """
        w = np.asarray(waveform, dtype=np.float64)
        if len(w) < 4:
            return np.zeros_like(w)
        w_ = np.diff(w - w[0])
        w_ = WaveformProcessor.smooth(w_ - w_[0], smooth_window)
        w_ = np.diff(w_ - w_[0])
        if len(w_) >= 20:
            w_[0:3] = np.mean(w_[0:20])
        elif len(w_) >= 3:
            w_[0:3] = np.mean(w_[0:len(w_)])
        w_ = WaveformProcessor.smooth(w_ - w_[0], smooth_window)
        w_ = w_ - w_[0]
        return np.concatenate([[0.0, 0.0], w_])

    @staticmethod
    def differentiate(waveform: np.ndarray, smooth_window: int) -> np.ndarray:
        """Compute the 2nd derivative of a spike waveform with smoothing.

        Similar to :meth:`smooth_and_differentiate` but without the
        intermediate smoothing step between the two ``diff`` operations,
        and enforces a minimum smooth window of 5. Direct port of MATLAB
        ``Differentiate.m``.

        Args:
            waveform: 1-D spike waveform (voltage vs. time).
            smooth_window: Window size for the final moving-average smoothing.

        Returns:
            Smoothed second derivative, same length as *waveform*.
        """
        w = np.asarray(waveform, dtype=np.float64)
        if len(w) < 4:
            return np.zeros_like(w)
        w_ = np.diff(w - w[0])
        w_ = np.diff(w_ - w_[0])
        if len(w_) >= 20:
            w_[0:3] = np.mean(w_[0:20])
        elif len(w_) >= 3:
            w_[0:3] = np.mean(w_[0:len(w_)])
        w_ = WaveformProcessor.smooth(w_ - w_[0], max(smooth_window, 5))
        w_ = w_ - w_[0]
        return np.concatenate([[0.0, 0.0], w_])


# Backwards-compatible aliases
smooth = WaveformProcessor.smooth
smooth_and_differentiate = WaveformProcessor.smooth_and_differentiate
differentiate = WaveformProcessor.differentiate
