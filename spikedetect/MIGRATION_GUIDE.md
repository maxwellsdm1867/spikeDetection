# Migration Guide: MATLAB to Python

This guide maps every MATLAB function in the spike detection codebase to its Python equivalent in the `spikedetect` package.

## Function Mapping

| MATLAB Function | Python Function | Import Path |
|---|---|---|
| `filterDataWithSpikes.m` | `filter_data()` | `spikedetect.pipeline.filtering` |
| `findSpikeLocations.m` | `find_spike_locations()` | `spikedetect.pipeline.peaks` |
| `dtw_WarpingDistance.m` | `dtw_warping_distance()` | `spikedetect.pipeline.dtw` |
| `getSquiggleDistanceFromTemplate.m` | `match_template()` | `spikedetect.pipeline.template` |
| `likelyInflectionPoint.m` | `likely_inflection_point()` | `spikedetect.pipeline.inflection` |
| `estimateSpikeTimeFromInflectionPoint.m` | `estimate_spike_times()` | `spikedetect.pipeline.inflection` |
| `thegoodthebadandtheweird.m` | `classify_spikes()` | `spikedetect.pipeline.classify` |
| `smoothAndDifferentiate.m` | `smooth_and_differentiate()` | `spikedetect.utils` |
| `Differentiate.m` | `differentiate()` | `spikedetect.utils` |
| `smooth()` (MATLAB builtin) | `smooth()` | `spikedetect.utils` |
| `spikeDetectionNonInteractive.m` | `detect_spikes()` | `spikedetect.pipeline.detect` |
| `spikeDetection.m` | `detect_spikes()` + GUI classes | `spikedetect.pipeline.detect` / `spikedetect.gui` |
| `filter_sliderGUI.m` | `FilterGUI` | `spikedetect.gui.filter_gui` |
| `spikeclickerGUI.m` / seed template | `TemplateSelectionGUI` | `spikedetect.gui.template_gui` |
| `spikeThresholdUpdateGUI.m` | `ThresholdGUI` | `spikedetect.gui.threshold_gui` |
| `spikeSpotCheck.m` | `SpotCheckGUI` | `spikedetect.gui.spotcheck_gui` |
| `cleanUpSpikeVarsStruct.m` | `SpikeDetectionParams.validate()` | `spikedetect.models` |
| `setacqpref/getacqpref` | `save_params()` / `load_params()` | `spikedetect.io.config` |
| `abfload.m` | `load_abf()` | `spikedetect.io.abf` |
| `filterMembraneVoltage.m` | Not ported (preprocessing) | — |
| `lowPassFilterMembraneVoltage.m` | Not ported (preprocessing) | — |

## Data Structure Mapping

| MATLAB | Python | Notes |
|---|---|---|
| `vars` struct (global) | `SpikeDetectionParams` dataclass | No global state; passed as argument |
| `trial` struct | `Recording` dataclass | |
| `trial.spikes` | `SpikeDetectionResult.spike_times` | 0-based indices (not 1-based) |
| `trial.spikeDetectionParams` | `SpikeDetectionResult.params` | |
| `trial.spikeSpotChecked` | `SpikeDetectionResult.spot_checked` | |

## Parameter Name Mapping

| MATLAB (`vars.`) | Python (`SpikeDetectionParams.`) |
|---|---|
| `fs` | `fs` |
| `spikeTemplateWidth` | `spike_template_width` |
| `hp_cutoff` | `hp_cutoff` |
| `lp_cutoff` | `lp_cutoff` |
| `diff` | `diff_order` |
| `peak_threshold` | `peak_threshold` |
| `Distance_threshold` | `distance_threshold` |
| `Amplitude_threshold` | `amplitude_threshold` |
| `spikeTemplate` | `spike_template` |
| `polarity` | `polarity` |
| `likelyiflpntpeak` | `likely_inflection_point_peak` |
| `lastfilename` | `last_filename` |

## Key Conversion Notes

### Indexing: 1-based to 0-based
All array indices in Python are 0-based. MATLAB spike times stored in `.mat` files are 1-based and are loaded as-is by `load_recording()`. New detections from `detect_spikes()` return 0-based indices.

### Filtering: Causal (not zero-phase)
The MATLAB code uses `filter()` (causal/forward-only). The Python port uses `scipy.signal.lfilter()`, NOT `filtfilt()`. This preserves the exact same phase behavior.

### DTW: Squared Euclidean cost
The DTW implementation uses squared Euclidean local cost `(r[i] - t[j])^2`, matching the original MATLAB code. Standard DTW libraries (dtaidistance, etc.) use absolute difference by default, so a custom implementation is used.

### Smoothing: `smooth()` equivalent
MATLAB's `smooth(x, n)` is a moving average with 'nearest' boundary handling. The Python equivalent is `scipy.ndimage.uniform_filter1d(x, size=n, mode='nearest')`.

### No global state
The MATLAB code relies heavily on `global vars`. The Python port eliminates all global state — parameters are passed explicitly to each function and returned as part of the result.

## Cross-Validation Status

The Python pipeline has been cross-validated against the original MATLAB code on a real 50 kHz recording (296 spikes). Every intermediate variable was compared at each pipeline stage. Full details in `CROSS_VALIDATION_REPORT.md`.

| Pipeline Stage | Agreement |
|---|---|
| Data preparation (trimming) | Exact |
| Butterworth filter coefficients | Exact (rtol=1e-12) |
| Filtered signal (HP, BP, diff+polarity) | Near-exact (max diff ~2e-15 V) |
| Peak detection (locations, count) | Exact (296/296 identical) |
| DTW distances | Near-exact (rtol=1e-6) |
| Amplitude projections | Near-exact (rtol=1e-6 with same inflection point) |
| Threshold mask | Exact (boolean match) |
| Spike time correction | Median 2-sample (40 us) jitter, max 11 samples (220 us) |

**The Python package is a drop-in replacement for the MATLAB pipeline.** The only measurable differences occur in the spike time correction step, where inherent numerical sensitivity in the smoothed 2nd-derivative peak-finding produces sub-millisecond timing jitter that would not affect any standard electrophysiological analysis.

### Known Differences

1. **Inflection point**: Python may differ by +-1 sample from MATLAB due to boundary handling differences between `scipy.ndimage.uniform_filter1d(mode='nearest')` and MATLAB `smooth()`. This causes a ~26 us systematic offset in corrected spike times — negligible for all downstream analyses.

2. **Spike time correction jitter**: The per-spike inflection point search (smooth -> diff -> smooth -> diff -> findpeaks) is inherently sensitive to floating-point precision. 93% of spikes match within +-100 us; the remaining 7% differ by up to 220 us. This jitter is a property of the algorithm, not a porting defect.

## Data Format Specification

For a complete specification of all data structures, file formats, and how to write a translator from any acquisition system, see **[DATA_FORMAT_SPEC.md](DATA_FORMAT_SPEC.md)**.

## Quick Start

```python
import spikedetect as sd

# Load a .mat file from the MATLAB pipeline
from spikedetect.io import load_recording
recording = load_recording("path/to/trial.mat")

# Use existing MATLAB parameters to re-detect
if recording.result:
    params = recording.result.params
    result = sd.detect_spikes(recording, params)
    print(f"Detected {result.n_spikes} spikes")

# Or start fresh
params = sd.SpikeDetectionParams.default(fs=recording.sample_rate)
params.spike_template = template_array  # from GUI or known template
params.hp_cutoff = 834.6
params.lp_cutoff = 160.2
params.diff_order = 1
params.polarity = -1
params.peak_threshold = 7.6e-5
params.distance_threshold = 9.6
params.amplitude_threshold = -0.08

result = sd.detect_spikes(recording, params)
```

## Interactive Workflow

```python
from spikedetect.gui import FilterGUI, TemplateSelectionGUI, ThresholdGUI, SpotCheckGUI

# 1. Tune filter parameters
filter_gui = FilterGUI(recording.voltage, params)
params = filter_gui.run()

# 2. Select seed template
template_gui = TemplateSelectionGUI(filtered_data, params)
params.spike_template = template_gui.run()

# 3. Run detection
result = sd.detect_spikes(recording, params)

# 4. Adjust thresholds
threshold_gui = ThresholdGUI(match_result, params)
params = threshold_gui.run()

# 5. Spot-check individual spikes
spotcheck = SpotCheckGUI(recording, result)
result = spotcheck.run()
```
