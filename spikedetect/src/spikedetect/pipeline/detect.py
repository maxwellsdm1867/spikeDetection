"""Full spike detection pipeline orchestrator.

Ports MATLAB ``spikeDetectionNonInteractive.m`` — runs the complete
detection pipeline: filter -> find peaks -> template match -> threshold ->
correct spike times via inflection point.
"""

from __future__ import annotations

import logging

import numpy as np

from spikedetect.models import Recording, SpikeDetectionParams, SpikeDetectionResult
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.inflection import InflectionPointDetector
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher

logger = logging.getLogger(__name__)


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
        pre_filter_cutoff: float | None = None,
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
        pre_filter_cutoff : float or None, optional
            If not None, apply a low-pass pre-filter at this cutoff (Hz) to
            the raw voltage before trimming and running the pipeline. Ports
            MATLAB ``lowPassFilterMembraneVoltage.m``. Default None (no
            pre-filtering).

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
        if not isinstance(recording, Recording):
            raise TypeError(
                f"Expected a Recording object, got {type(recording).__name__}. "
                "Load your data first with load_recording('file.mat') or create "
                "a Recording(name='...', voltage=array, sample_rate=10000)."
            )
        if not isinstance(params, SpikeDetectionParams):
            raise TypeError(
                f"Expected SpikeDetectionParams, got {type(params).__name__}. "
                "Create params with SpikeDetectionParams(fs=10000) or "
                "SpikeDetectionParams.default(fs=10000)."
            )

        import copy
        params = copy.copy(params)
        params = params.validate()

        if params.spike_template is None:
            raise ValueError(
                "No spike template provided. Use the interactive GUI "
                "(FilterGUI / TemplateSelectionGUI) to select one, or pass "
                "a 1-D numpy array as params.spike_template."
            )

        voltage = recording.voltage.copy()
        fs = params.fs

        # Optional low-pass pre-filter (MATLAB lowPassFilterMembraneVoltage)
        if pre_filter_cutoff is not None:
            logger.info("Applying pre-filter: cutoff=%.0f Hz", pre_filter_cutoff)
            voltage = SignalFilter.pre_filter(voltage, fs, cutoff=pre_filter_cutoff)
        duration_sec = len(voltage) / fs
        logger.info(
            "Starting spike detection on '%s' (%.1f s, %.0f Hz)",
            recording.name, duration_sec, fs,
        )

        # Trim start (MATLAB: start_point = round(.01*fs))
        start_point = round(start_offset * fs)
        stop_point = len(voltage)
        unfiltered_data = voltage[start_point:stop_point]

        # Step 1: Filter
        logger.info(
            "Filtering: hp=%.0f Hz, lp=%.0f Hz, diff_order=%d, polarity=%+d",
            params.hp_cutoff, params.lp_cutoff, params.diff_order, params.polarity,
        )
        filtered_data = SignalFilter.filter_data(
            unfiltered_data,
            fs=fs,
            hp_cutoff=params.hp_cutoff,
            lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order,
            polarity=params.polarity,
        )

        # Step 2: Find peaks
        # Guard against absurd peak_threshold (use local var, don't mutate params)
        peak_thresh = params.peak_threshold
        if peak_thresh > 1e4 * np.std(filtered_data):
            peak_thresh = 3 * np.std(filtered_data)

        spike_locs = PeakFinder.find_spike_locations(
            filtered_data,
            peak_threshold=peak_thresh,
            fs=fs,
            spike_template_width=params.spike_template_width,
        )

        logger.info("Found %d candidate peaks", len(spike_locs))

        if len(spike_locs) == 0:
            logger.info("No candidate peaks found -- returning empty result")
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
        logger.info("Computing DTW template match for %d candidates...", len(spike_locs))
        match_result = TemplateMatcher.match(
            spike_locs,
            params.spike_template,
            filtered_data,
            unfiltered_data,
            params.spike_template_width,
            fs,
        )

        if len(match_result.dtw_distances) == 0:
            logger.info("No valid candidates after template matching")
            return SpikeDetectionResult(
                spike_times=np.array([], dtype=np.int64),
                spike_times_uncorrected=np.array([], dtype=np.int64),
                params=params,
            )

        # Step 4: Threshold — keep spikes with DTW < threshold AND amplitude > threshold
        # Use match_result.spike_locs (not original spike_locs) since NaN
        # candidates may have been filtered out during template matching.
        suspect = (match_result.dtw_distances < params.distance_threshold) & (
            match_result.amplitudes > params.amplitude_threshold
        )
        accepted_locs = match_result.spike_locs[suspect]

        logger.info(
            "Accepted %d / %d candidates (distance < %.1f, amplitude > %.3f)",
            len(accepted_locs), len(match_result.spike_locs),
            params.distance_threshold, params.amplitude_threshold,
        )

        if len(accepted_locs) == 0:
            logger.info("No spikes passed thresholds -- returning empty result")
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

        logger.info(
            "Detection complete: %d spikes found in '%s'",
            len(corrected_global), recording.name,
        )

        return SpikeDetectionResult(
            spike_times=corrected_global.astype(np.int64),
            spike_times_uncorrected=uncorrected_global.astype(np.int64),
            params=params,
        )


# Backwards-compatible alias
detect_spikes = SpikeDetector.detect
