"""Full spike detection pipeline orchestrator.

Ports MATLAB ``spikeDetectionNonInteractive.m`` — runs the complete
detection pipeline: filter -> find peaks -> template match -> threshold ->
correct spike times via inflection point.
"""

from __future__ import annotations

import numpy as np

from spikedetect.models import Recording, SpikeDetectionParams, SpikeDetectionResult
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.inflection import InflectionPointDetector
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher


class SpikeDetector:
    """Full non-interactive spike detection pipeline.

    Ports MATLAB ``spikeDetectionNonInteractive.m``. Runs the complete
    pipeline: filter -> find peaks -> template match -> threshold ->
    correct spike times via inflection point.

    Examples
    --------
    >>> result = SpikeDetector.detect(recording, params)
    >>> print(result.n_spikes)
    """

    @staticmethod
    def detect(
        recording: Recording,
        params: SpikeDetectionParams,
        start_offset: float = 0.01,
    ) -> SpikeDetectionResult:
        """Run the full non-interactive spike detection pipeline.

        Parameters
        ----------
        recording : Recording
            The electrophysiology recording to analyze.
        params : SpikeDetectionParams
            Detection parameters. Must have ``spike_template`` set
            (non-None) for detection to proceed.
        start_offset : float, optional
            Fraction of recording to skip at the start (default 0.01 = 1%).
            Matches MATLAB ``start_point = round(.01*fs)``.

        Returns
        -------
        SpikeDetectionResult
            Detection results including corrected and uncorrected spike times
            (as 0-based sample indices into the *original* recording voltage).

        Raises
        ------
        ValueError
            If ``params.spike_template`` is None.

        Notes
        -----
        Original MATLAB function: ``spikeDetectionNonInteractive.m``

        The returned spike times are offset by ``start_point`` so they index
        into the original ``recording.voltage`` array, matching the MATLAB
        convention ``trial.spikes = spikes_detected + start_point``.
        """
        params = params.validate()

        if params.spike_template is None:
            raise ValueError(
                "spike_template must be set before running detection. "
                "Use the interactive GUI or provide a template array."
            )

        voltage = recording.voltage.copy()
        fs = params.fs

        # Trim start (MATLAB: start_point = round(.01*fs))
        start_point = round(start_offset * fs)
        stop_point = len(voltage)
        unfiltered_data = voltage[start_point:stop_point]

        # Step 1: Filter
        filtered_data = SignalFilter.filter_data(
            unfiltered_data,
            fs=fs,
            hp_cutoff=params.hp_cutoff,
            lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order,
            polarity=params.polarity,
        )

        # Step 2: Find peaks
        # Guard against absurd peak_threshold
        if params.peak_threshold > 1e4 * np.std(filtered_data):
            params.peak_threshold = 3 * np.std(filtered_data)

        spike_locs = PeakFinder.find_spike_locations(
            filtered_data,
            peak_threshold=params.peak_threshold,
            fs=fs,
            spike_template_width=params.spike_template_width,
        )

        if len(spike_locs) == 0:
            return SpikeDetectionResult(
                spike_times=np.array([], dtype=np.int64),
                spike_times_uncorrected=np.array([], dtype=np.int64),
                params=params,
            )

        # Verify no duplicates
        assert len(spike_locs) == len(np.unique(spike_locs)), (
            "Duplicate peak locations detected"
        )

        # Step 3: Template matching (DTW distance + amplitude)
        match_result = TemplateMatcher.match(
            spike_locs,
            params.spike_template,
            filtered_data,
            unfiltered_data,
            params.spike_template_width,
            fs,
        )

        if len(match_result.dtw_distances) == 0:
            return SpikeDetectionResult(
                spike_times=np.array([], dtype=np.int64),
                spike_times_uncorrected=np.array([], dtype=np.int64),
                params=params,
            )

        # Step 4: Threshold — keep spikes with DTW < threshold AND amplitude > threshold
        suspect = (match_result.dtw_distances < params.distance_threshold) & (
            match_result.amplitudes > params.amplitude_threshold
        )
        accepted_locs = spike_locs[suspect]

        if len(accepted_locs) == 0:
            return SpikeDetectionResult(
                spike_times=np.array([], dtype=np.int64),
                spike_times_uncorrected=np.array([], dtype=np.int64),
                params=params,
            )

        # Step 5: Correct spike times using inflection point
        uf_cands = match_result.unfiltered_candidates[:, suspect]
        waveforms = uf_cands - uf_cands[0:1, :]

        corrected, uncorrected, infl_peak = InflectionPointDetector.estimate_spike_times(
            accepted_locs,
            waveforms,
            match_result.dtw_distances[suspect],
            params.spike_template_width,
            fs,
            params.distance_threshold,
            params.likely_inflection_point_peak,
        )

        # Update params with computed inflection point
        params.likely_inflection_point_peak = infl_peak

        # Offset back to original recording indices
        corrected_global = corrected + start_point
        uncorrected_global = uncorrected + start_point

        return SpikeDetectionResult(
            spike_times=corrected_global.astype(np.int64),
            spike_times_uncorrected=uncorrected_global.astype(np.int64),
            params=params,
        )


# Backwards-compatible alias
detect_spikes = SpikeDetector.detect
