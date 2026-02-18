"""Inflection point detection and spike time correction.

Ports MATLAB ``likelyInflectionPoint.m`` and ``estimateSpikeTimeFromInflectionPoint.m``.
Finds the peak of the smoothed second derivative to determine precise spike timing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from spikedetect.utils import WaveformProcessor


class InflectionPointDetector:
    """Detect inflection points and correct spike timing.

    Ports MATLAB ``likelyInflectionPoint.m`` and
    ``estimateSpikeTimeFromInflectionPoint.m``.

    Examples
    --------
    >>> peak, deriv = InflectionPointDetector.likely_inflection_point(
    ...     waveforms, distances, stw=51, fs=10000)
    >>> corrected, uncorrected, peak = InflectionPointDetector.estimate_spike_times(
    ...     locs, waveforms, distances, stw=51, fs=10000, dist_thresh=10.0)
    """

    @staticmethod
    def likely_inflection_point(
        spike_waveforms: np.ndarray,
        dtw_distances: np.ndarray,
        spike_template_width: int,
        fs: float,
    ) -> tuple[int, np.ndarray]:
        """Find the likely inflection point from the best spike candidates.

        Parameters
        ----------
        spike_waveforms : np.ndarray, shape (n_window, n_candidates)
            Unfiltered waveform windows for all candidates.
        dtw_distances : np.ndarray, shape (n_candidates,)
            DTW distances for each candidate.
        spike_template_width : int
            Template width in samples.
        fs : float
            Sample rate in Hz.

        Returns
        -------
        inflection_peak : int
            0-based index of the inflection point within the spike window.
        spike_waveform_2nd_deriv : np.ndarray
            The smoothed 2nd derivative of the average spike waveform.

        Notes
        -----
        Original MATLAB function: ``likelyInflectionPoint.m``
        """
        stw = spike_template_width
        idx_i = round(stw / 6)
        idx_f = round(stw / 24)
        idx_m = round(stw * 4 / 5)

        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half
        smth_start = round(fs / 2000)

        valid = dtw_distances > 0
        if np.sum(valid) == 0:
            return idx_m, np.zeros(len(spike_window))

        valid_dists = dtw_distances[valid]
        q25 = np.quantile(valid_dists, 0.25)
        good_spikes = dtw_distances < q25

        if np.sum(good_spikes) < 4:
            if len(dtw_distances) == 1:
                good_spikes = np.array([True])
            else:
                order = np.argsort(dtw_distances)
                cnt = 0
                while np.sum(good_spikes) < min(len(good_spikes) // 2, 4) and cnt < len(order):
                    good_spikes[order[cnt]] = True
                    cnt += 1

        good_wfs = spike_waveforms[:, good_spikes]
        spike_waveform = np.nanmean(good_wfs, axis=1)
        spike_waveform = spike_waveform - np.min(spike_waveform)
        wf_max = np.max(spike_waveform)
        if wf_max > 0:
            spike_waveform = spike_waveform / wf_max

        smth_w_small = max(round(fs / 4000), 1)
        smth_w = max(round(fs / 2000), 1)
        spike_waveform = WaveformProcessor.smooth(spike_waveform - spike_waveform[0], smth_w_small)
        spike_waveform_2d = WaveformProcessor.smooth_and_differentiate(spike_waveform, smth_w)

        if smth_start < len(spike_waveform_2d):
            spike_waveform_2d = spike_waveform_2d - spike_waveform_2d[smth_start]

        region = spike_waveform_2d[idx_i:len(spike_waveform_2d) - idx_f]
        if len(region) > 0:
            rmin = np.min(region)
            rmax = np.max(region)
            if rmax != rmin:
                spike_waveform_2d = (spike_waveform_2d - rmin) / (rmax - rmin)

        search_region = spike_waveform_2d[idx_i + 1:len(spike_waveform_2d) - idx_f]
        min_prominence = 0.014 * 251 / stw

        if len(search_region) > 0:
            peaks, properties = find_peaks(search_region, prominence=min_prominence)
            pks = properties["prominences"] if len(peaks) > 0 else np.array([])
        else:
            peaks = np.array([], dtype=int)
            pks = np.array([])

        if len(peaks) == 0:
            expn = 1
            while len(peaks) == 0 and expn < 20:
                if len(search_region) > 0:
                    peaks, properties = find_peaks(
                        search_region, prominence=min_prominence / (2**expn)
                    )
                    pks = properties.get("prominences", np.array([]))
                expn += 1

        if len(peaks) == 0:
            return idx_m, spike_waveform_2d

        infl_peaks = peaks + idx_i + 1
        peak_heights = spike_waveform_2d[infl_peaks] if len(infl_peaks) > 0 else pks

        dist_to_mid = np.abs(infl_peaks - idx_m)
        min_dist = np.min(dist_to_mid)
        closest_mask = dist_to_mid == min_dist
        candidate_peaks = infl_peaks[closest_mask]
        candidate_heights = peak_heights[closest_mask]

        if len(candidate_peaks) > 1:
            best_idx = np.argmax(candidate_heights)
            inflection_peak = int(candidate_peaks[best_idx])
        else:
            inflection_peak = int(candidate_peaks[0])

        return inflection_peak, spike_waveform_2d

    @staticmethod
    def estimate_spike_times(
        spike_locs: np.ndarray,
        spike_waveforms: np.ndarray,
        dtw_distances: np.ndarray,
        spike_template_width: int,
        fs: float,
        distance_threshold: float,
        likely_iflpnt_peak: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Correct spike times using inflection point of 2nd derivative.

        Parameters
        ----------
        spike_locs : np.ndarray, shape (n_spikes,)
            Uncorrected spike locations (0-based sample indices).
        spike_waveforms : np.ndarray, shape (n_window, n_spikes)
            Unfiltered waveform windows for each spike.
        dtw_distances : np.ndarray, shape (n_spikes,)
            DTW distances for each spike.
        spike_template_width : int
            Template width in samples.
        fs : float
            Sample rate in Hz.
        distance_threshold : float
            DTW distance threshold for spike acceptance.
        likely_iflpnt_peak : int or None
            Pre-computed inflection point index.

        Returns
        -------
        corrected_locs : np.ndarray
        uncorrected_locs : np.ndarray
        inflection_peak : int

        Notes
        -----
        Original MATLAB function: ``estimateSpikeTimeFromInflectionPoint.m``
        """
        stw = spike_template_width

        inflection_peak, _ = InflectionPointDetector.likely_inflection_point(
            spike_waveforms, dtw_distances, stw, fs
        )
        if likely_iflpnt_peak is not None:
            inflection_peak = likely_iflpnt_peak

        half = stw // 2
        window = np.arange(-half, half + 1)
        spike_window = window - half

        uncorrected = spike_locs.copy()
        corrected = spike_locs.copy()
        smth_w = max(round(fs / 2000), 1)

        start_idx = round(fs / 10000 * 20)
        end_idx = round(fs / 10000 * 6)

        for i in range(len(corrected)):
            if dtw_distances[i] > distance_threshold:
                comparison = np.abs(corrected - uncorrected[i])
                if np.sum(comparison < stw) > 1:
                    continue

            waveform = spike_waveforms[:, i].copy()
            waveform_smoothed = WaveformProcessor.smooth(waveform - waveform[0], smth_w)
            waveform_2d = WaveformProcessor.smooth_and_differentiate(waveform_smoothed, smth_w)

            region_s = waveform_smoothed[start_idx + 1:len(waveform_smoothed) - end_idx]
            region_d = waveform_2d[start_idx + 1:len(waveform_2d) - end_idx]

            if len(region_s) > 0:
                s_min, s_max = np.min(region_s), np.max(region_s)
                if s_max != s_min:
                    waveform_smoothed = (waveform_smoothed - s_min) / (s_max - s_min)

            if len(region_d) > 0:
                d_min, d_max = np.min(region_d), np.max(region_d)
                if d_max != d_min:
                    waveform_2d = (waveform_2d - d_min) / (d_max - d_min)

            search = waveform_2d[start_idx + 1:len(waveform_2d) - end_idx]
            min_prom = 0.04 * 251 / stw

            if len(search) > 0:
                peaks, _ = find_peaks(search, prominence=min_prom)
            else:
                peaks = np.array([], dtype=int)

            infl_peak = peaks + start_idx + 1

            if len(infl_peak) > 1:
                dist_to_expected = np.abs(infl_peak - inflection_peak)
                best = np.argmin(dist_to_expected)
                infl_peak = np.array([infl_peak[best]])

            min_peak_pos = round(fs / 10000 * 30)
            if (
                len(infl_peak) == 1
                and infl_peak[0] > min_peak_pos
                and infl_peak[0] < len(waveform_2d) - end_idx
            ):
                corrected[i] = corrected[i] + spike_window[infl_peak[0]]
            else:
                corrected[i] = corrected[i] + spike_window[inflection_peak]

        # Handle duplicate spike times
        _, unique_idx = np.unique(corrected, return_index=True)
        duplicate_mask = np.ones(len(corrected), dtype=bool)
        duplicate_mask[unique_idx] = False
        duplicate_indices = np.where(duplicate_mask)[0]

        for idx in duplicate_indices:
            dup_val = corrected[idx]
            rep_indices = np.where(corrected == dup_val)[0]
            for j, ridx in enumerate(rep_indices):
                corrected[ridx] = corrected[ridx] + j

        if len(np.unique(corrected)) != len(corrected):
            import warnings
            warnings.warn("Still some duplicate spike values after correction.")

        return corrected, uncorrected, inflection_peak


# Backwards-compatible aliases
likely_inflection_point = InflectionPointDetector.likely_inflection_point
estimate_spike_times = InflectionPointDetector.estimate_spike_times
