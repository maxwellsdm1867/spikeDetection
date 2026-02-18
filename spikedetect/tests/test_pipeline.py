"""End-to-end pipeline tests for spikedetect."""

import numpy as np
import pytest

from spikedetect.models import SpikeDetectionResult
from spikedetect.pipeline.detect import detect_spikes


class TestDetectSpikes:
    @pytest.fixture
    def pipeline_recording(self):
        """Recording designed to survive the filter+diff pipeline.

        Instead of embedding raw templates in noise, we create a signal
        that produces clear peaks after bandpass filtering + differentiation.
        The template is extracted from the same trimmed+filtered signal that
        detect_spikes will use internally.
        """
        from spikedetect.models import Recording, SpikeDetectionParams
        from spikedetect.pipeline.filtering import filter_data

        fs = 10000
        duration = 5.0
        n_samples = int(fs * duration)
        rng = np.random.default_rng(42)

        noise = rng.normal(0, 0.001, n_samples)

        # Embed sharp voltage transients that will survive filtering
        true_positions = np.array([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
        for pos in true_positions:
            t_local = np.arange(-25, 26) / fs
            spike = 5.0 * np.exp(-0.5 * (t_local / 0.0003) ** 2)
            noise[pos - 25:pos + 26] += spike

        params = SpikeDetectionParams(fs=fs, hp_cutoff=200, lp_cutoff=800, diff_order=1, polarity=1)

        # Simulate what detect_spikes does: trim start then filter
        start_point = round(0.01 * fs)  # = 100
        unfiltered_data = noise[start_point:]
        filtered = filter_data(unfiltered_data, fs, params.hp_cutoff, params.lp_cutoff, params.diff_order, params.polarity)

        # Extract template at the first spike position (adjusted for trim)
        stw = params.spike_template_width
        half = stw // 2
        spike_in_trimmed = true_positions[0] - start_point
        template = filtered[spike_in_trimmed - half:spike_in_trimmed + half + 1].copy()

        params.spike_template = template
        params.peak_threshold = 0.005
        params.amplitude_threshold = 0.001

        recording = Recording(name="pipeline_test", voltage=noise, sample_rate=fs)
        return recording, params, true_positions

    def test_detects_embedded_spikes(self, pipeline_recording):
        recording, params, true_positions = pipeline_recording
        result = detect_spikes(recording, params)

        assert isinstance(result, SpikeDetectionResult)
        assert result.n_spikes > 0

    def test_corrected_times_near_true_positions(self, pipeline_recording):
        recording, params, true_positions = pipeline_recording
        result = detect_spikes(recording, params)

        if result.n_spikes == 0:
            pytest.skip("No spikes detected; cannot verify positions")

        tolerance = 100  # samples (10ms at 10kHz)
        matched = 0
        for detected in result.spike_times:
            min_dist = np.min(np.abs(true_positions - detected))
            if min_dist < tolerance:
                matched += 1

        assert matched >= 1, f"Only {matched} out of {result.n_spikes} detected spikes matched true positions"

    def test_returns_spike_detection_result(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params)
        assert isinstance(result, SpikeDetectionResult)
        assert hasattr(result, "spike_times")
        assert hasattr(result, "spike_times_uncorrected")
        assert hasattr(result, "params")

    def test_raises_without_template(self):
        from spikedetect.models import Recording, SpikeDetectionParams

        recording = Recording(name="test", voltage=np.zeros(10000), sample_rate=10000)
        params = SpikeDetectionParams(fs=10000)
        with pytest.raises(ValueError, match="No spike template provided"):
            detect_spikes(recording, params)

    def test_empty_result_for_silent_recording(self, default_params):
        from spikedetect.models import Recording

        silent = Recording(
            name="silent",
            voltage=np.zeros(50000),
            sample_rate=10000,
        )
        result = detect_spikes(silent, default_params)
        assert result.n_spikes == 0

    def test_spike_times_are_int64(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params)
        assert result.spike_times.dtype == np.int64
        assert result.spike_times_uncorrected.dtype == np.int64
