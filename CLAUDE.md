# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python package (`spikedetect`) for detecting and sorting neural/EMG spikes from electrophysiology recordings. Ported from a MATLAB codebase using DTW template-matching with amplitude thresholds to classify candidate spikes, with interactive Matplotlib GUIs for parameter tuning and manual spot-checking.

The original MATLAB code is preserved in `legacy/` for reference.

## Installation and Dependencies

The Python package lives in the `spikedetect/` subdirectory. All install commands must be run from there.

### Prerequisites

- **Python >= 3.9** (check: `python --version`)
- **pip** (check: `pip --version`)
- If the user doesn't have Python, recommend [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Install Commands

```bash
# IMPORTANT: cd into the spikedetect/ subdirectory first
cd spikedetect

# Basic install (core detection pipeline)
pip install -e .

# With development tools (for running tests)
pip install -e ".[dev]"

# With ABF and HDF5 file support
pip install -e ".[io]"

# With numba acceleration for faster DTW
pip install -e ".[fast]"

# Everything at once
pip install -e ".[all]"
```

### Required Dependencies (auto-installed by pip)

| Package | Min Version | What it's for |
|---------|-------------|---------------|
| numpy | >= 1.24 | Array operations, all numerical computation |
| scipy | >= 1.10 | Butterworth filters (`scipy.signal`), peak finding, .mat file I/O |
| matplotlib | >= 3.7 | Plotting, all interactive GUIs |

### Optional Dependencies

| Extra | Packages | When to install |
|-------|----------|-----------------|
| `[io]` | `pyabf >= 2.3`, `h5py >= 3.8` | User has `.abf` files (Axon Binary Format) or MATLAB v7.3 HDF5 `.mat` files |
| `[fast]` | `numba >= 0.57`, `dtaidistance >= 2.3` | Large datasets where DTW speed matters. Falls back to pure numpy if not installed. |
| `[dev]` | `pytest >= 7.3`, `pytest-cov >= 4.0`, `ruff >= 0.1` | Running tests, linting |
| `[all]` | All of the above | Full install |

### Optional User Dependencies (not in pyproject.toml)

| Package | When needed |
|---------|-------------|
| `pandas` | Only if user calls `result.to_dataframe()` to export spike times as a DataFrame |
| `ipympl` | Only for Jupyter notebook GUI interaction (`%matplotlib widget` backend) |

### Common Installation Issues

1. **numba/numpy version mismatch**: If the user sees `_ARRAY_API not found` or `compiled using NumPy 1.x cannot be run in NumPy 2.x` warnings — this is harmless, the pipeline falls back to pure numpy. Fix options:
   - `pip uninstall numba` (remove numba, pure numpy is fine)
   - `pip install numpy<2` (downgrade numpy)
   - `pip install numba --upgrade` (upgrade numba to match numpy 2)

2. **h5py not installed**: If loading a MATLAB v7.3 .mat file fails with `ImportError: h5py`, run `pip install h5py>=3.8` or `pip install -e ".[io]"`

3. **pyabf not installed**: If loading a .abf file fails with `ImportError: pyabf`, run `pip install pyabf>=2.3` or `pip install -e ".[io]"`

4. **Jupyter GUIs not interactive**: If GUI widgets don't respond in Jupyter, the user needs: `pip install ipympl` and `%matplotlib widget` at the top of the notebook

5. **Package not found after install**: Make sure install was run from `spikedetect/` subdirectory (not the repo root). The `pyproject.toml` is at `spikedetect/pyproject.toml`.

### Verify Installation

```bash
# Quick check
python -c "import spikedetect; print(spikedetect.__version__)"
# Should print: 0.1.0

# Full verification (run tests)
cd spikedetect && python -m pytest tests/ -v
# Should show: 79 passed
```

## Build and Test Commands

```bash
# IMPORTANT: run from the spikedetect/ subdirectory
cd spikedetect

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_dtw.py -v

# Run a specific test
python -m pytest tests/test_pipeline.py::TestDetectSpikes::test_detects_embedded_spikes -v

# Run with coverage
python -m pytest tests/ --cov=spikedetect --cov-report=term-missing
```

## Architecture

### Package Structure

```
spikedetect/src/spikedetect/
├── __init__.py          # Public API: all classes + detect_spikes alias
├── models.py            # SpikeDetectionParams, Recording, SpikeDetectionResult dataclasses
├── utils.py             # WaveformProcessor (smooth, smooth_and_differentiate, differentiate)
├── io/
│   ├── config.py        # JSON param persistence (~/.spikedetect/)
│   ├── mat.py           # .mat loading (v5/v7 via scipy, v7.3 via h5py)
│   ├── abf.py           # ABF loading via pyabf
│   └── native.py        # HDF5 native format via h5py
├── pipeline/
│   ├── filtering.py     # SignalFilter — Butterworth bandpass + differentiation
│   ├── peaks.py         # PeakFinder — find candidate spike peaks
│   ├── dtw.py           # DTW — squared-Euclidean cost DTW (optional numba JIT)
│   ├── template.py      # TemplateMatcher — DTW template matching + amplitude projection
│   ├── inflection.py    # InflectionPointDetector — 2nd derivative spike time correction
│   ├── classify.py      # SpikeClassifier — quantile-based 4-category classification
│   └── detect.py        # SpikeDetector — full pipeline orchestrator
└── gui/
    ├── filter_gui.py    # FilterGUI — interactive filter parameter tuning
    ├── template_gui.py  # TemplateSelectionGUI — click to select seed spikes
    ├── threshold_gui.py # ThresholdGUI — DTW/amplitude scatter threshold adjustment
    └── spotcheck_gui.py # SpotCheckGUI — spike-by-spike review
```

### Detection Pipeline (SpikeDetector.detect)

1. **SignalFilter.filter_data** — Causal Butterworth bandpass (`scipy.signal.lfilter`, NOT `filtfilt`) with optional differentiation and polarity flip
2. **PeakFinder.find_spike_locations** — `scipy.signal.find_peaks` with height and distance constraints
3. **TemplateMatcher.match** — Extract windows, normalize, compute DTW distance against template, compute amplitude via projection
4. **Threshold** — Accept spikes where `DTW_distance < threshold` AND `amplitude > threshold`
5. **InflectionPointDetector.estimate_spike_times** — Correct timing using peak of smoothed 2nd derivative

### Class-Based Design

All pipeline functions are organized as static methods within classes. Each module also exports backwards-compatible function aliases (e.g., `filter_data = SignalFilter.filter_data`) so both styles work:

```python
# Class-based (preferred)
from spikedetect import SignalFilter, SpikeDetector
filtered = SignalFilter.filter_data(voltage, fs=10000, hp_cutoff=200, lp_cutoff=800)

# Function alias (backwards-compatible)
from spikedetect.pipeline.filtering import filter_data
filtered = filter_data(voltage, fs=10000, hp_cutoff=200, lp_cutoff=800)
```

### Critical Porting Details

- MATLAB `smooth(x, n)` → `scipy.ndimage.uniform_filter1d(x, size=n, mode='nearest')`
- MATLAB `filter(b, a, x)` → `scipy.signal.lfilter(b, a, x)` (causal, NOT `filtfilt`)
- DTW uses squared-Euclidean cost `(r[i] - t[j])^2`, not absolute difference
- MATLAB 1-based indexing converted to Python 0-based throughout
- `SpikeDetectionParams` dataclass replaces the MATLAB global `vars` struct
- JSON persistence in `~/.spikedetect/` replaces MATLAB `setacqpref`/`getacqpref`

### Test Data

`LEDFlashTriggerPiezoControl_Raw_240430_F1_C1_5.mat` — Real recording (50kHz, 400k samples, 296 MATLAB-detected spikes) used as golden standard for cross-validation in `test_io.py`.

## Directory Layout

- `spikedetect/` — Python package (installable via pip)
- `legacy/` — Original MATLAB source code (functions/, scripts/, utils/, matlab_src/)
- `LICENSE` — Project license
