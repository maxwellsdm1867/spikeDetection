"""Bandpass filtering and optional differentiation for spike detection.

Ports the MATLAB function ``filterDataWithSpikes.m``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)


class SignalFilter:
    """Causal Butterworth bandpass filter with optional differentiation.

    Ports MATLAB ``filterDataWithSpikes.m``. All methods are static.

    Examples
    --------
    >>> filtered = SignalFilter.filter_data(voltage, fs=10000,
    ...     hp_cutoff=200, lp_cutoff=800, diff_order=1, polarity=-1)
    """

    @staticmethod
    def filter_data(
        unfiltered_data: np.ndarray,
        fs: float,
        hp_cutoff: float,
        lp_cutoff: float,
        diff_order: int = 0,
        polarity: int = 1,
    ) -> np.ndarray:
        """Apply bandpass filter with optional differentiation and polarity flip.

        Parameters
        ----------
        unfiltered_data : np.ndarray
            Raw 1-D voltage trace.
        fs : float
            Sampling rate in Hz.
        hp_cutoff : float
            High-pass filter cutoff frequency in Hz.
        lp_cutoff : float
            Low-pass filter cutoff frequency in Hz.
        diff_order : int, optional
            Derivative order applied after filtering (0, 1, or 2). Default 0.
        polarity : int, optional
            Multiply output by this (+1 or -1). Default 1.

        Returns
        -------
        np.ndarray
            Filtered 1-D float64 array, same length as input.

        Notes
        -----
        Uses ``scipy.signal.lfilter`` (causal), NOT ``filtfilt``.
        Original MATLAB function: ``filterDataWithSpikes.m``
        """
        data = np.asarray(unfiltered_data, dtype=np.float64).ravel()
        if len(data) == 0:
            return data

        wn_hp = hp_cutoff / (fs / 2.0)
        b_hp, a_hp = butter(3, wn_hp, btype="high")
        filtered_high = lfilter(b_hp, a_hp, data - data[0])

        wn_lp = lp_cutoff / (fs / 2.0)
        b_lp, a_lp = butter(3, wn_lp, btype="low")
        filtered = lfilter(b_lp, a_lp, filtered_high)

        if diff_order == 0:
            diff_filt = filtered
        elif diff_order == 1:
            diff_filt = np.empty_like(filtered)
            diff_filt[0] = 0.0
            diff_filt[1:] = np.diff(filtered)
            diff_filt[:100] = 0.0
        elif diff_order == 2:
            diff_filt = np.empty_like(filtered)
            diff_filt[0] = 0.0
            diff_filt[1] = 0.0
            diff_filt[2:] = np.diff(filtered, n=2)
            diff_filt[:100] = 0.0
        else:
            raise ValueError(
                f"diff_order must be 0, 1, or 2, got {diff_order}. "
                "Use 0 for no differentiation, 1 for first derivative "
                "(recommended), or 2 for second derivative."
            )

        return (polarity * diff_filt).astype(np.float64)


    @staticmethod
    def pre_filter(
        voltage: np.ndarray,
        fs: float,
        cutoff: float = 3000.0,
        order: int = 12,
    ) -> np.ndarray:
        """Apply a low-pass pre-filter to raw voltage before spike detection.

        Ports MATLAB ``lowPassFilterMembraneVoltage.m``. Subtracts the first
        sample (DC baseline) before filtering and adds it back afterwards,
        matching the MATLAB ``filter(d1, v - base) + base`` pattern.

        Parameters
        ----------
        voltage : np.ndarray
            Raw 1-D voltage trace.
        fs : float
            Sampling rate in Hz.
        cutoff : float, optional
            Low-pass cutoff frequency in Hz. Default 3000.
        order : int, optional
            Butterworth filter order. Default 12 (matches MATLAB ``designfilt``).

        Returns
        -------
        np.ndarray
            Low-pass filtered 1-D float64 array, same length as input.
        """
        data = np.asarray(voltage, dtype=np.float64).ravel()
        if len(data) == 0:
            return data

        wn = cutoff / (fs / 2.0)
        b, a = butter(order, wn, btype="low")
        base = data[0]
        filtered = lfilter(b, a, data - base) + base
        return filtered.astype(np.float64)


# Backwards-compatible aliases
filter_data = SignalFilter.filter_data
pre_filter = SignalFilter.pre_filter
