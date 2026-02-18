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
