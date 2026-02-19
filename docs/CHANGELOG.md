# Changelog

## v0.1.0 -- Initial Python Release

Python refactor of the [MATLAB spikeDetection](https://github.com/tony-azevedo/spikeDetection) codebase. All detection algorithms produce numerically identical results to the original MATLAB code (verified: 296/296 spike match on real recording data).

### New in Python vs MATLAB

- **No global state** -- parameters are passed explicitly as `SpikeDetectionParams`, not via MATLAB `global vars`
- **Pip-installable** -- `pip install -e .` with optional extras for numba acceleration and ABF/HDF5 I/O
- **Class-based API** -- all pipeline stages are classes with static methods (`SignalFilter`, `PeakFinder`, `TemplateMatcher`, etc.) plus backwards-compatible function aliases
- **135 automated tests** covering all pipeline stages, I/O, cross-validation against MATLAB, and end-to-end detection
- **Cross-validated** against MATLAB output on real electrophysiology data (see `CROSS_VALIDATION_REPORT.md` for 18-test step-by-step comparison of every intermediate variable)
- **Data format specification** (`DATA_FORMAT_SPEC.md`) for writing translators from any acquisition frontend
- **Google Python Style Guide** compliance (docstrings, line length, naming)

---

## Quality-of-Life Improvements

These improvements were added after the initial port to make the package more usable for scientists who are not CS majors. **Zero numerical behavior was changed** -- all pipeline computations remain identical to the MATLAB original.

### Better Error Messages

All error messages now include:
- **What went wrong** (the specific value that failed validation)
- **Why it's wrong** (the constraint that was violated)
- **How to fix it** (actionable suggestion)

Before (MATLAB-style):
```
ValueError: HP cutoff must be below Nyquist
```

After:
```
ValueError: High-pass cutoff (6000 Hz) must be below the Nyquist frequency
(5000.0 Hz). Try lowering hp_cutoff or using a higher sample rate.
```

Full list of improved messages:

| Situation | Error message |
|-----------|---------------|
| Negative sample rate | "Sample rate (fs) must be positive, got -1. Check that the recording was loaded correctly." |
| HP cutoff too high | "High-pass cutoff (6000 Hz) must be below the Nyquist frequency (5000.0 Hz). Try lowering hp_cutoff or using a higher sample rate." |
| LP cutoff too high | "Low-pass cutoff (6000 Hz) must be below the Nyquist frequency (5000.0 Hz). Try lowering lp_cutoff or using a higher sample rate." |
| Bad diff order | "diff_order must be 0, 1, or 2, got 3. Use 0 for no differentiation, 1 for first derivative (recommended), or 2 for second derivative." |
| Bad polarity | "polarity must be -1 or 1, got 0. Use 1 for upward spikes, -1 for downward spikes." |
| 2-D template | "spike_template must be a 1-D array, got shape (51, 3). Flatten it with template.ravel() before passing." |
| Empty voltage | "Voltage array is empty. Check that the recording file loaded correctly." |
| No template | "No spike template provided. Use the interactive GUI (FilterGUI / TemplateSelectionGUI) to select one, or pass a 1-D numpy array as params.spike_template." |
| Wrong type for recording | "Expected a Recording object, got ndarray. Load your data first with load_recording('file.mat') or create a Recording(name='...', voltage=array, sample_rate=10000)." |
| Wrong type for params | "Expected SpikeDetectionParams, got dict. Create params with SpikeDetectionParams(fs=10000) or SpikeDetectionParams.default(fs=10000)." |
| Wrong file extension | "For ABF files, use load_abf() instead" |

### Pipeline Logging

The detection pipeline now logs progress at each step using Python's `logging` module. Enable with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Output during `detect_spikes()`:

```
INFO:spikedetect.pipeline.detect:Starting spike detection on 'trial_001' (8.0 s, 50000 Hz)
INFO:spikedetect.pipeline.detect:Filtering: hp=800 Hz, lp=160 Hz, diff_order=1, polarity=-1
INFO:spikedetect.pipeline.detect:Found 342 candidate peaks
INFO:spikedetect.pipeline.detect:Computing DTW template match for 342 candidates...
INFO:spikedetect.pipeline.detect:Accepted 296 / 342 candidates (distance < 9.6, amplitude > -0.080)
INFO:spikedetect.pipeline.detect:Detection complete: 296 spikes found in 'trial_001'
```

Use `logging.DEBUG` for additional detail (edge-excluded peaks, boundary-skipped candidates).

### Convenience Methods

#### `Recording.plot(show_spikes=True)`
Plot the voltage trace with optional spike markers. Returns a matplotlib Figure.

```python
rec.result = result
fig = rec.plot()
```

#### `Recording.duration` and `Recording.n_samples`
Properties for quick info without manual calculation.

```python
print(f"{rec.duration:.2f} seconds, {rec.n_samples} samples")
```

#### `SpikeDetectionResult.summary()`
Human-readable text summary with spike count, time range, mean ISI, firing rate, and spot-check status.

```python
print(result.summary())
# Spike Detection Result
#   Spikes found: 296
#   Time range: 0.162 - 7.882 s
#   Mean ISI: 26.2 ms (range 5.3 - 112.4 ms)
#   Mean firing rate: 38.2 Hz
#   Spot-checked: no
```

#### `SpikeDetectionResult.to_dataframe()`
Export spike times to a pandas DataFrame for analysis. Requires pandas (optional).

```python
df = result.to_dataframe()
df.to_csv("spike_times.csv", index=False)
```

#### `SpikeDetectionParams.default(fs)`
Smart defaults that auto-scale filter cutoffs for low sample rates to stay below Nyquist.

```python
params = sd.SpikeDetectionParams.default(fs=2000)
# hp_cutoff and lp_cutoff are automatically scaled down
```

### Input Validation

Added validation at system boundaries to catch mistakes early:

- **Recording**: Rejects empty voltage arrays and non-positive sample rates
- **SpikeDetectionParams**: Validates 1-D template shape on construction
- **detect_spikes()**: Type-checks that arguments are `Recording` and `SpikeDetectionParams` with actionable error messages
- **load_recording()**: Validates file extension, gives specific guidance for ABF files

### Top-Level Imports

I/O functions are now available directly from `import spikedetect`:

```python
import spikedetect as sd

rec = sd.load_recording("trial.mat")    # no need to import from sd.io.mat
rec = sd.load_abf("recording.abf")     # no need to import from sd.io.abf
sd.save_result("output.mat", rec)      # no need to import from sd.io.mat
```

---

## Bug Fixes (Code Review)

These bugs were found by independent code review and fixed before release:

| Severity | File | Issue | Fix |
|----------|------|-------|-----|
| CRITICAL | `utils.py` | `smooth_and_differentiate` crashes on waveforms < 4 samples | Added early-return guard for short waveforms |
| CRITICAL | `template.py` / `detect.py` | `spike_locs` array misalignment after NaN removal in template matching | Added `spike_locs` to `TemplateMatchResult` dataclass; `detect.py` uses filtered locations |
| HIGH | `detect.py` | `detect_spikes()` mutates the caller's `params` object | Uses `copy.copy(params)` internally |
| HIGH | `detect.py` | `peak_threshold` guard modifies `params` permanently | Uses local variable `peak_thresh` |
| MEDIUM | `classify.py` | `len(bad)` checks mask length instead of count of True values | Changed to `np.sum(bad)` |
| MEDIUM | `template.py` | `idx_f = round(stw/24)` could be 0 for small templates | Changed to `max(1, round(stw/24))` |
| MEDIUM | `filtering.py` | No handling for empty input arrays | Added early return for empty input |
| LOW | `dtw.py` | numba import fails with numpy 2.x | Catches any exception during import |
