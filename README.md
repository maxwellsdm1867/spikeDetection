# spikedetect

Spike detection and sorting for electrophysiology recordings using DTW template matching.

A Python refactor of the MATLAB [spikeDetection](https://github.com/tony-azevedo/spikeDetection/tree/master) codebase by Tony Azevedo. Detects neural/EMG spikes by bandpass filtering, peak finding, Dynamic Time Warping (DTW) template matching, and inflection-point timing correction. Includes interactive Matplotlib GUIs for filter tuning, template selection, threshold adjustment, and spike-by-spike review.

**Author**: Arthur Hong
**Original MATLAB code**: [tony-azevedo/spikeDetection](https://github.com/tony-azevedo/spikeDetection)

## Installation

```bash
# Basic install
cd spikedetect
pip install -e .

# With all optional dependencies (numba acceleration, ABF/HDF5 I/O, dev tools)
pip install -e ".[all]"

# Individual extras
pip install -e ".[fast]"   # numba + dtaidistance for faster DTW
pip install -e ".[io]"     # pyabf + h5py for ABF and HDF5 files
pip install -e ".[dev]"    # pytest + ruff for development
```

**Requirements**: Python >= 3.9, numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7

## Quick Start

### Batch Detection (Non-Interactive)

```python
import spikedetect as sd
from spikedetect.io.mat import load_recording

# Load a recording from a .mat file
rec, params, existing_spikes = load_recording("path/to/trial.mat")

# Run detection with loaded parameters
result = sd.detect_spikes(rec, params)
print(f"Detected {result.n_spikes} spikes")
print(f"Spike times (seconds): {result.spike_times_seconds[:10]}")
```

### Fresh Detection with Custom Parameters

```python
import numpy as np
import spikedetect as sd

# Create a recording from raw voltage data
recording = sd.Recording(
    name="my_recording",
    voltage=voltage_array,    # 1-D numpy array
    sample_rate=50000,        # Hz
)

# Configure detection parameters
params = sd.SpikeDetectionParams(
    fs=50000,
    hp_cutoff=800,            # High-pass cutoff (Hz)
    lp_cutoff=160,            # Low-pass cutoff (Hz)
    diff_order=1,             # Differentiation order (0, 1, or 2)
    polarity=-1,              # Signal polarity flip
    peak_threshold=7.6e-5,    # Peak detection sensitivity
    distance_threshold=9.6,   # DTW distance threshold
    amplitude_threshold=-0.08,# Minimum spike amplitude
    spike_template=template,  # Template waveform (1-D array)
)

result = sd.detect_spikes(recording, params)
```

### Interactive Workflow (GUI)

```python
from spikedetect.gui import FilterGUI, TemplateSelectionGUI, ThresholdGUI, SpotCheckGUI

# 1. Tune filter parameters with sliders
filter_gui = FilterGUI(recording.voltage, params)
params = filter_gui.run()

# 2. Click on peaks to select a seed template
template_gui = TemplateSelectionGUI(filtered_data, params)
params.spike_template = template_gui.run()

# 3. Run detection
result = sd.detect_spikes(recording, params)

# 4. Adjust DTW/amplitude thresholds interactively
threshold_gui = ThresholdGUI(match_result, params)
params = threshold_gui.run()

# 5. Review individual spikes (y/n/arrow keys)
spotcheck = SpotCheckGUI(recording, result)
result = spotcheck.run()  # result.spot_checked = True
```

For Jupyter notebooks, use `%matplotlib widget` (requires ipympl).

## Detection Pipeline

The pipeline runs these steps in order:

| Step | Class | Description |
|------|-------|-------------|
| 1. Filter | `SignalFilter` | Causal Butterworth bandpass + optional differentiation |
| 2. Find peaks | `PeakFinder` | Locate candidate spike peaks above threshold |
| 3. Template match | `TemplateMatcher` | DTW distance + amplitude projection against template |
| 4. Threshold | — | Accept spikes with low DTW distance and high amplitude |
| 5. Correct timing | `InflectionPointDetector` | Refine spike times using 2nd derivative inflection |

Additional utilities:
- `DTW` — Dynamic Time Warping with squared-Euclidean cost (optional numba JIT)
- `SpikeClassifier` — Quantile-based classification into good/weird/weirdbad/bad
- `WaveformProcessor` — Smoothing and differentiation helpers

## Using Individual Pipeline Components

Each pipeline stage is a class with static methods, usable independently:

```python
from spikedetect import SignalFilter, PeakFinder, DTW, TemplateMatcher

# Filter raw data
filtered = SignalFilter.filter_data(voltage, fs=50000, hp_cutoff=800, lp_cutoff=160,
                                     diff_order=1, polarity=-1)

# Find candidate peaks
locs = PeakFinder.find_spike_locations(filtered, peak_threshold=7e-5,
                                        fs=50000, spike_template_width=251)

# Compute DTW distance between two waveforms
distance, warped_r, warped_t = DTW.warping_distance(waveform_a, waveform_b)

# Match all candidates against a template
result = TemplateMatcher.match(locs, template, filtered, unfiltered,
                                spike_template_width=251, fs=50000)
```

## File I/O

```python
from spikedetect.io.mat import load_recording, save_result
from spikedetect.io.abf import load_abf
from spikedetect.io.native import save_native, load_native
from spikedetect.io.config import save_params, load_params

# MATLAB .mat files (v5, v7, and v7.3/HDF5)
rec, params, spikes = load_recording("trial.mat")

# ABF files (requires pyabf)
rec = load_abf("recording.abf")

# Native HDF5 format (compact, with gzip compression)
save_native("output.h5", recording)
rec = load_native("output.h5")

# Save/load detection parameters as JSON (~/.spikedetect/)
save_params(params, "experiment_001")
params = load_params("experiment_001")
```

## Project Structure

```
spikeDetection/
├── README.md
├── LICENSE
├── CLAUDE.md
├── spikedetect/              # Python package
│   ├── pyproject.toml
│   ├── MIGRATION_GUIDE.md    # MATLAB-to-Python function mapping
│   ├── src/spikedetect/
│   │   ├── models.py         # SpikeDetectionParams, Recording, SpikeDetectionResult
│   │   ├── utils.py          # WaveformProcessor
│   │   ├── io/               # File I/O (mat, abf, hdf5, config)
│   │   ├── pipeline/         # Detection pipeline modules
│   │   └── gui/              # Interactive Matplotlib GUIs
│   └── tests/                # 79 pytest tests
└── legacy/                   # Original MATLAB source code
```

## Testing

```bash
cd spikedetect

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_dtw.py -v

# Run a single test
python -m pytest tests/test_pipeline.py::TestDetectSpikes::test_detects_embedded_spikes -v

# With coverage
python -m pytest tests/ --cov=spikedetect --cov-report=term-missing
```

## Migrating from MATLAB

See [MIGRATION_GUIDE.md](spikedetect/MIGRATION_GUIDE.md) for a complete mapping of MATLAB functions, data structures, and parameter names to their Python equivalents.

Key differences:
- **No global state** — parameters are passed explicitly, not via `global vars`
- **0-based indexing** — all spike times are 0-based sample indices
- **Causal filtering** — uses `lfilter` (not `filtfilt`) to match MATLAB `filter()`
- **Squared-Euclidean DTW** — custom implementation matching the original MATLAB cost metric

## License

MIT License. Copyright (c) 2020 Anthony Azevedo, 2026 Arthur Hong. See [LICENSE](LICENSE).
