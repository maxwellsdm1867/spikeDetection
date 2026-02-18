"""Template matching for spike candidates (ports getSquiggleDistanceFromTemplate.m).

Extracts waveform windows around each candidate peak location, normalizes
them, computes DTW distance against the spike template, and computes spike
amplitude using a projection method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spikedetect.pipeline.dtw import dtw_warping_distance
from spikedetect.pipeline.inflection import likely_inflection_point


@dataclass
class TemplateMatchResult:
    """Results from template matching against all candidate peaks.

    Attributes
    ----------
    unfiltered_candidates : np.ndarray, shape (n_window, n_candidates)
        Unfiltered waveform windows around each candidate.
    filtered_candidates : np.ndarray, shape (n_window, n_candidates)
        Filtered waveform windows around each candidate.
    norm_filtered_candidates : np.ndarray, shape (n_window, n_candidates)
        Min-max normalized filtered waveform windows.
    dtw_distances : np.ndarray, shape (n_candidates,)
        DTW distance for each candidate.
    amplitudes : np.ndarray, shape (n_candidates,)
        Projected amplitude for each candidate.
    window : np.ndarray
        Template window offsets (centered on peak).
    spike_window : np.ndarray
        Extended spike window offsets.
    likely_inflection_peak : int
        Estimated inflection point index within the spike window.
    """

    unfiltered_candidates: np.ndarray
    filtered_candidates: np.ndarray
    norm_filtered_candidates: np.ndarray
    dtw_distances: np.ndarray
    amplitudes: np.ndarray
    window: np.ndarray
    spike_window: np.ndarray
    likely_inflection_peak: int


class TemplateMatcher:
    """Template matching for spike candidates using DTW distance.

    Ports MATLAB ``getSquiggleDistanceFromTemplate.m``. Extracts waveform
    windows around each candidate peak, normalizes them, computes DTW
    distance against a spike template, and computes amplitude via projection.

    Examples
    --------
    >>> result = TemplateMatcher.match(spike_locs, template, filtered,
    ...     unfiltered, stw=51, fs=10000)
    """

    @staticmethod
    def _empty_result(stw: int) -> TemplateMatchResult:
        """Return an empty result when there are no spike locations."""
        window = np.arange(-stw // 2, stw // 2 + 1)
        spike_window = window - stw // 2
        return TemplateMatchResult(
            unfiltered_candidates=np.empty((0, 0)),
            filtered_candidates=np.empty((0, 0)),
            norm_filtered_candidates=np.empty((0, 0)),
            dtw_distances=np.empty(0),
            amplitudes=np.empty(0),
            window=window,
            spike_window=spike_window,
            likely_inflection_peak=0,
        )

    @staticmethod
    def match(
        spike_locs: np.ndarray,
        spike_template: np.ndarray,
        filtered_data: np.ndarray,
        unfiltered_data: np.ndarray,
        spike_template_width: int,
        fs: float,
    ) -> TemplateMatchResult:
        """Match candidate spike waveforms against a template using DTW.

        Parameters
        ----------
        spike_locs : np.ndarray, shape (n_candidates,)
            Peak locations (0-based sample indices) in the filtered data.
        spike_template : np.ndarray, shape (spike_template_width,)
            Reference spike template waveform.
        filtered_data : np.ndarray, shape (n_samples,)
            Bandpass-filtered signal.
        unfiltered_data : np.ndarray, shape (n_samples,)
            Original unfiltered signal.
        spike_template_width : int
            Template half-width in samples.
        fs : float
            Sample rate in Hz.

        Returns
        -------
        result : TemplateMatchResult
            All matching results.

        Notes
        -----
        Original MATLAB function: ``getSquiggleDistanceFromTemplate.m``
        """
        stw = spike_template_width

        if len(spike_locs) == 0:
            return TemplateMatcher._empty_result(stw)

        # Window definitions matching MATLAB:
        #   window = -floor(stw/2): floor(stw/2)
        #   spikewindow = window - floor(stw/2)
        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half
        n_window = len(window)
        n_spike_window = len(spike_window)
        n_candidates = len(spike_locs)

        # Normalize template (min-max)
        t_min = np.min(spike_template)
        t_max = np.max(spike_template)
        norm_template = (spike_template - t_min) / (t_max - t_min) if t_max != t_min else spike_template * 0

        # Pre-allocate candidate arrays
        dtw_distances = np.zeros(n_candidates, dtype=np.float64)
        uf_candidates = np.full((n_spike_window, n_candidates), np.nan, dtype=np.float64)
        f_candidates = np.full((n_window, n_candidates), np.nan, dtype=np.float64)
        norm_f_candidates = np.full((n_window, n_candidates), np.nan, dtype=np.float64)

        for i, loc in enumerate(spike_locs):
            # Boundary check matching MATLAB (uses float stw/2):
            upper = min(loc + stw / 2, len(filtered_data))
            lower = max(loc - stw / 2, 0)
            if upper - lower < stw:
                continue

            # Verify window indices are valid
            f_indices = loc + window
            uf_indices = loc + spike_window
            if f_indices[0] < 0 or f_indices[-1] >= len(filtered_data):
                continue
            if uf_indices[0] < 0 or uf_indices[-1] >= len(unfiltered_data):
                continue

            # Extract filtered window (centered on peak)
            cur_spike = filtered_data[f_indices]
            f_candidates[:, i] = cur_spike

            # Extract unfiltered window (shifted by -half)
            uf_candidates[:, i] = unfiltered_data[uf_indices]

            # Normalize and compute DTW distance
            c_min = np.min(cur_spike)
            c_max = np.max(cur_spike)
            if c_max != c_min:
                norm_cur = (cur_spike - c_min) / (c_max - c_min)
            else:
                norm_cur = np.zeros_like(cur_spike)
            norm_f_candidates[:, i] = norm_cur

            dist, _, _ = dtw_warping_distance(norm_cur, norm_template)
            dtw_distances[i] = dist

        # Compute inflection point and amplitude projection
        inflection_peak, _ = likely_inflection_point(
            uf_candidates, dtw_distances, stw, fs
        )

        # Amplitude projection matching MATLAB
        idx_f = round(stw / 24)

        # Get the average spike waveform for amplitude calculation
        good_mask = dtw_distances < np.quantile(dtw_distances[dtw_distances > 0], 0.25) if np.sum(dtw_distances > 0) > 0 else np.ones(n_candidates, dtype=bool)
        good_waveforms = uf_candidates[:, good_mask & ~np.isnan(uf_candidates[0, :])]
        if good_waveforms.shape[1] > 0:
            spike_waveform = np.nanmean(good_waveforms, axis=1)
            spike_waveform = spike_waveform - np.min(spike_waveform)
            if np.max(spike_waveform) > 0:
                spike_waveform = spike_waveform / np.max(spike_waveform)
        else:
            spike_waveform = np.zeros(n_spike_window)

        # Compute projection vector s_hat
        ip = inflection_peak
        end_idx = n_spike_window - idx_f
        if ip < end_idx and ip >= 0:
            s_hat = spike_waveform[ip:end_idx] - spike_waveform[ip]
            s_sum = np.sum(s_hat)
            if s_sum != 0:
                s_hat = s_hat / s_sum
            else:
                s_hat = np.ones_like(s_hat) / len(s_hat)

            # Project each candidate
            segment = uf_candidates[ip:end_idx, :] - uf_candidates[ip:ip + 1, :]
            amplitudes = segment.T @ s_hat
        else:
            amplitudes = np.zeros(n_candidates)

        # Remove candidates that couldn't be fully extracted (NaN from edge proximity)
        valid_mask = ~np.any(np.isnan(uf_candidates), axis=0)
        if not np.all(valid_mask):
            uf_candidates = uf_candidates[:, valid_mask]
            f_candidates = f_candidates[:, valid_mask]
            norm_f_candidates = norm_f_candidates[:, valid_mask]
            dtw_distances = dtw_distances[valid_mask]
            amplitudes = amplitudes[valid_mask]

        return TemplateMatchResult(
            unfiltered_candidates=uf_candidates,
            filtered_candidates=f_candidates,
            norm_filtered_candidates=norm_f_candidates,
            dtw_distances=dtw_distances,
            amplitudes=amplitudes,
            window=window,
            spike_window=spike_window,
            likely_inflection_peak=inflection_peak,
        )


# Backwards-compatible alias
match_template = TemplateMatcher.match
