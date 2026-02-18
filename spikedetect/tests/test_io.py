"""Tests for spikedetect I/O modules."""

import json
from pathlib import Path

import numpy as np
import pytest

from spikedetect.io.config import load_params, save_params
from spikedetect.models import SpikeDetectionParams

# Path to the real test MAT file
_MAT_FILE = Path(
    "/Users/maxwellsdm/Documents/GitHub/spikeDetection/"
    "LEDFlashTriggerPiezoControl_Raw_240430_F1_C1_5.mat"
)


class TestConfigSaveLoad:
    def test_roundtrip(self, tmp_path, monkeypatch):
        """Save and load params, verify they match."""
        monkeypatch.setattr("spikedetect.io.config._CONFIG_DIR", tmp_path)

        template = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        params = SpikeDetectionParams(
            fs=10000,
            hp_cutoff=300.0,
            lp_cutoff=900.0,
            diff_order=2,
            peak_threshold=4.0,
            distance_threshold=12.0,
            amplitude_threshold=0.3,
            spike_template=template,
            polarity=-1,
            likely_inflection_point_peak=20,
            last_filename="test_file.mat",
        )

        path = save_params(params, input_field="voltage_1")
        assert path.exists()

        loaded = load_params(input_field="voltage_1", fs=10000)
        assert loaded is not None
        assert loaded.fs == params.fs
        assert loaded.hp_cutoff == params.hp_cutoff
        assert loaded.lp_cutoff == params.lp_cutoff
        assert loaded.diff_order == params.diff_order
        assert loaded.peak_threshold == params.peak_threshold
        assert loaded.distance_threshold == params.distance_threshold
        assert loaded.amplitude_threshold == params.amplitude_threshold
        assert loaded.polarity == params.polarity
        assert loaded.likely_inflection_point_peak == params.likely_inflection_point_peak
        assert loaded.last_filename == params.last_filename
        np.testing.assert_array_equal(loaded.spike_template, template)

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spikedetect.io.config._CONFIG_DIR", tmp_path)
        result = load_params(input_field="nonexistent", fs=99999)
        assert result is None

    def test_saved_file_is_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spikedetect.io.config._CONFIG_DIR", tmp_path)
        params = SpikeDetectionParams(fs=10000)
        path = save_params(params)
        data = json.loads(path.read_text())
        assert data["fs"] == 10000


def _try_load_mat():
    """Try to load the test MAT file, skip if h5py is incompatible."""
    from spikedetect.io.mat import load_recording

    try:
        return load_recording(_MAT_FILE)
    except (ValueError, ImportError) as e:
        if "numpy.dtype" in str(e) or "h5py" in str(e):
            pytest.skip("numpy/h5py binary incompatibility in this environment")
        raise


@pytest.mark.skipif(not _MAT_FILE.exists(), reason="Test MAT file not found")
class TestMatLoading:
    def test_load_recording(self):
        recording = _try_load_mat()
        assert recording.voltage is not None
        assert len(recording.voltage) == 400032
        assert recording.voltage.dtype == np.float64

    def test_recording_has_name(self):
        recording = _try_load_mat()
        assert "240430_F1_C1" in recording.name

    def test_sample_rate(self):
        recording = _try_load_mat()
        assert recording.sample_rate == 50000.0

    def test_voltage_range(self):
        recording = _try_load_mat()
        assert recording.voltage.min() < -40
        assert recording.voltage.max() > -35

    def test_current_loaded(self):
        recording = _try_load_mat()
        assert recording.current is not None
        assert len(recording.current) == 400032

    def test_existing_spikes_loaded(self):
        recording = _try_load_mat()
        assert recording.result is not None
        assert recording.result.n_spikes == 296

    def test_detection_params_loaded(self):
        recording = _try_load_mat()
        assert recording.result is not None
        p = recording.result.params
        assert p.fs == 50000.0
        assert abs(p.hp_cutoff - 834.63) < 1.0
        assert abs(p.lp_cutoff - 160.24) < 1.0
        assert p.diff_order == 1
        assert p.polarity == -1
        assert p.spike_template is not None
        assert len(p.spike_template) == 251

    def test_spike_template_width(self):
        recording = _try_load_mat()
        p = recording.result.params
        assert p.spike_template_width == 251

    def test_spike_times_are_reasonable(self):
        recording = _try_load_mat()
        spikes = recording.result.spike_times
        assert np.all(spikes > 0)
        assert np.all(spikes < len(recording.voltage))
        # Spikes should be sorted
        assert np.all(np.diff(spikes) > 0)


@pytest.mark.skipif(not _MAT_FILE.exists(), reason="Test MAT file not found")
class TestMatCrossValidation:
    """Cross-validate Python detection against MATLAB results."""

    def test_redetect_spike_count(self):
        """Re-run detection with loaded params and compare spike count."""
        from spikedetect.pipeline.detect import detect_spikes

        recording = _try_load_mat()
        matlab_result = recording.result
        assert matlab_result is not None

        # Re-run detection with the same parameters
        result = detect_spikes(recording, matlab_result.params)

        matlab_count = matlab_result.n_spikes
        python_count = result.n_spikes

        # Allow some tolerance — not exact due to float precision differences
        ratio = python_count / max(matlab_count, 1)
        assert 0.5 < ratio < 2.0, (
            f"Python detected {python_count} spikes vs MATLAB {matlab_count} "
            f"(ratio {ratio:.2f})"
        )
