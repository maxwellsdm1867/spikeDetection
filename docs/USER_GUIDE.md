# spikedetect User Guide

Complete reference for the `spikedetect` Python package -- spike detection and sorting for electrophysiology recordings.

If you're new, start with the [Getting Started guide](GETTING_STARTED.md) first.

---

## Table of Contents

1. [How the Pipeline Works](#how-the-pipeline-works)
2. [Parameters Reference](#parameters-reference)
3. [Using Individual Pipeline Stages](#using-individual-pipeline-stages)
4. [Interactive GUIs](#interactive-guis)
5. [File I/O](#file-io)
6. [Batch Processing Multiple Recordings](#batch-processing-multiple-recordings)
7. [Understanding the Output](#understanding-the-output)
8. [Tips for Getting Good Results](#tips-for-getting-good-results)
9. [API Quick Reference](#api-quick-reference)

---

## How the Pipeline Works

`spikedetect` finds spikes in five steps. Each step is a separate class, and they are chained together by the `SpikeDetector.detect()` orchestrator.

```
Raw voltage
    |
    v
[1. SignalFilter]  -- Butterworth bandpass + optional differentiation
    |
    v
[2. PeakFinder]    -- Find candidate peaks above threshold
    |
    v
[3. TemplateMatcher] -- DTW distance + amplitude for each candidate
    |
    v
[4. Threshold]     -- Keep candidates with low DTW distance + high amplitude
    |
    v
[5. InflectionPointDetector] -- Correct spike timing via 2nd derivative
    |
    v
Spike times (sample indices)
```

### Step 1: Bandpass Filtering (`SignalFilter`)

Applies a 3rd-order Butterworth high-pass filter, then a 3rd-order low-pass filter to isolate the spike frequency band. Optionally differentiates the signal (1st or 2nd derivative) and flips polarity.

Key detail: Uses **causal filtering** (`scipy.signal.lfilter`), not zero-phase filtering (`filtfilt`). This matches the original MATLAB `filter()` function behavior. Causal filtering introduces a small phase delay, but this is corrected later by the inflection point timing correction.

### Step 2: Peak Finding (`PeakFinder`)

Finds local maxima in the filtered signal using `scipy.signal.find_peaks`. Peaks must be:
- Above `mean(signal) + peak_threshold`
- At least `fs / 1800` samples apart (prevents detecting the same spike twice)
- Far enough from the recording edges to extract a full template window

### Step 3: Template Matching (`TemplateMatcher`)

For each candidate peak:
1. Extracts a waveform window centered on the peak
2. Min-max normalizes the waveform to [0, 1]
3. Computes Dynamic Time Warping (DTW) distance against the spike template
4. Computes spike amplitude using a projection method based on the inflection point

The DTW uses **squared-Euclidean** local cost `(a[i] - b[j])^2`, not absolute difference. This matches the original MATLAB implementation and produces different results than standard DTW libraries.

### Step 4: Thresholding

Keeps candidates where:
- `DTW_distance < distance_threshold` (shape matches template)
- `amplitude > amplitude_threshold` (large enough to be a spike)

### Step 5: Inflection Point Correction (`InflectionPointDetector`)

Refines each spike's timing by finding the inflection point (the steepest rise) on the unfiltered waveform. This corrects for the phase delay introduced by causal filtering.

Uses the peak of the smoothed 2nd derivative, looking near the expected `likely_inflection_point_peak` location.

---

## Parameters Reference

All detection parameters are stored in `SpikeDetectionParams`. Here is every field with its meaning, range, and how to tune it.

### Filter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fs` | float | (required) | Sample rate in Hz. Must match your recording. |
| `hp_cutoff` | float | 200.0 | High-pass filter cutoff in Hz. Removes slow drifts. Increase to isolate fast transients. Must be < `fs/2`. |
| `lp_cutoff` | float | 800.0 | Low-pass filter cutoff in Hz. Removes high-frequency noise. Decrease for cleaner but blunter spikes. Must be < `fs/2`. |
| `diff_order` | int | 1 | Differentiation after filtering. **0** = none, **1** = first derivative (recommended -- enhances spike peaks), **2** = second derivative (sharper but noisier). |
| `polarity` | int | 1 | Multiply signal by this value. Use **+1** for upward spikes, **-1** for downward spikes. |

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `peak_threshold` | float | 5.0 | Peak detection sensitivity. Candidates must exceed `mean(filtered) + peak_threshold`. Lower = more candidates (more sensitive). Higher = fewer candidates (more selective). |
| `spike_template` | ndarray or None | None | The reference spike waveform. Must be set before detection. Obtained from `TemplateSelectionGUI` or a previous run. 1-D numpy array. |
| `spike_template_width` | int | auto | Template half-width in samples. Auto-computed as `round(0.005 * fs) + 1` if not set. Updated to match template length during validation. |
| `distance_threshold` | float | 15.0 | Maximum DTW distance to accept a candidate. Lower = stricter (fewer false positives, may miss unusual spikes). Higher = looser (catches more spikes, more false positives). |
| `amplitude_threshold` | float | 0.2 | Minimum spike amplitude (projection value). Lower = catches smaller spikes. Can be negative for certain signal polarities. |

### Internal Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `likely_inflection_point_peak` | int or None | None | Expected inflection point index within the spike window. Auto-computed during detection. Rarely needs manual setting. |
| `last_filename` | str | "" | Name of the last file processed. Used for tracking. |

### Creating Parameters

```python
import spikedetect as sd

# Auto-scaled defaults (recommended)
params = sd.SpikeDetectionParams.default(fs=50000)

# Manual creation with all fields
params = sd.SpikeDetectionParams(
    fs=50000,
    hp_cutoff=800,
    lp_cutoff=160,
    diff_order=1,
    polarity=-1,
    peak_threshold=7.6e-5,
    distance_threshold=9.6,
    amplitude_threshold=-0.08,
    spike_template=template_array,
)

# Validate (checks ranges, updates computed fields)
params = params.validate()
```

---

## Using Individual Pipeline Stages

Each pipeline stage is a class with static methods. You can use them independently for custom workflows.

### Filtering

```python
from spikedetect import SignalFilter

filtered = SignalFilter.filter_data(
    voltage,          # 1-D numpy array
    fs=50000,         # sample rate
    hp_cutoff=800,    # high-pass cutoff Hz
    lp_cutoff=160,    # low-pass cutoff Hz
    diff_order=1,     # 0, 1, or 2
    polarity=-1,      # +1 or -1
)
```

### Peak Finding

```python
from spikedetect import PeakFinder

locs = PeakFinder.find_spike_locations(
    filtered,                  # filtered signal
    peak_threshold=7.6e-5,     # minimum peak height
    fs=50000,                  # sample rate
    spike_template_width=251,  # template width in samples
)
# locs is a 1-D int64 array of sample indices
```

### DTW Distance

```python
from spikedetect import DTW

distance, warped_a, warped_b = DTW.warping_distance(waveform_a, waveform_b)
# distance: float (accumulated squared-Euclidean cost)
# warped_a, warped_b: aligned versions of the inputs
```

### Template Matching

```python
from spikedetect import TemplateMatcher

result = TemplateMatcher.match(
    spike_locs=locs,           # from PeakFinder
    spike_template=template,   # 1-D template waveform
    filtered_data=filtered,    # filtered signal
    unfiltered_data=voltage,   # raw voltage
    spike_template_width=251,  # template width
    fs=50000,                  # sample rate
)
# result.dtw_distances: DTW distance per candidate
# result.amplitudes: amplitude per candidate
# result.spike_locs: locations (may be shorter than input due to edge removal)
```

### Spike Classification

```python
from spikedetect import SpikeClassifier

good, weird, weirdbad, bad = SpikeClassifier.classify(
    distances=result.dtw_distances,
    amplitudes=result.amplitudes,
    distance_threshold=9.6,
    amplitude_threshold=-0.08,
)
# Each is a boolean mask over candidates
```

### Waveform Processing

```python
from spikedetect import WaveformProcessor

# Smooth and differentiate a waveform
smoothed = WaveformProcessor.smooth_and_differentiate(waveform, stw=251, fs=50000)

# Just differentiate
deriv = WaveformProcessor.differentiate(waveform)
```

---

## Interactive GUIs

Four interactive tools for visual parameter tuning. All use Matplotlib and work in both standalone Python and Jupyter notebooks.

### FilterGUI -- Tune filter parameters

Sliders for HP cutoff, LP cutoff, peak threshold. Radio buttons for diff order. Button for polarity toggle. Shows the filtered signal updating in real time.

```python
from spikedetect.gui import FilterGUI

filter_gui = FilterGUI(voltage, params)
params = filter_gui.run()  # blocks until you close the window
# params is updated with your chosen filter settings
```

### TemplateSelectionGUI -- Select a spike template

Displays the filtered signal with detected peaks marked. Click on peaks to select seed spikes. The GUI averages your selections (with cross-correlation alignment) to build a template.

```python
from spikedetect.gui import TemplateSelectionGUI

template_gui = TemplateSelectionGUI(filtered_data, params)
template = template_gui.run()
params.spike_template = template
```

### ThresholdGUI -- Adjust acceptance thresholds

Scatter plot of DTW distance vs. amplitude for all candidates. Click to move the threshold lines. Color-coded waveform panels show what's accepted/rejected. Press `b` to toggle between adjusting distance and amplitude thresholds.

```python
from spikedetect.gui import ThresholdGUI

threshold_gui = ThresholdGUI(match_result, params)
params = threshold_gui.run()
```

### SpotCheckGUI -- Review spikes one by one

Step through each detected spike. Keyboard controls:
- **y** / **Enter**: Accept this spike
- **n**: Reject this spike
- **Left/Right arrows**: Navigate between spikes
- **Tab**: Jump to the next unreviewed spike

Shows the unfiltered waveform, mean waveform overlay, 2nd derivative, and context window.

```python
from spikedetect.gui import SpotCheckGUI

spotcheck = SpotCheckGUI(recording, result)
result = spotcheck.run()
# result.spot_checked is now True
```

### Jupyter Notebooks

For Jupyter, add this at the top of your notebook:

```python
%matplotlib widget
```

This requires `ipympl`: `pip install ipympl`

---

## File I/O

### MATLAB .mat files

Supports MATLAB v5, v7, and v7.3 (HDF5) formats. Automatically detects the format.

```python
import spikedetect as sd

# Load a trial .mat file
rec = sd.load_recording("trial.mat")
# rec.name, rec.voltage, rec.sample_rate, rec.current (if available)
# rec.result (SpikeDetectionResult if the file had previous detection results)

# Save results back
from spikedetect.io.mat import save_result
save_result("trial_updated.mat", rec, result)
```

### ABF files (Axon Binary Format)

Requires `pyabf`: `pip install pyabf`

```python
rec = sd.load_abf("recording.abf")
```

### Native HDF5 format

Compact format with gzip compression. Good for long-term storage.

```python
from spikedetect.io.native import save_native, load_native

save_native("output.h5", recording)
rec = load_native("output.h5")
```

### Parameter persistence

Save and load detection parameters as JSON files in `~/.spikedetect/`.

```python
from spikedetect.io.config import save_params, load_params

save_params(params, "my_experiment")
params = load_params("my_experiment")
```

---

## Batch Processing Multiple Recordings

```python
import spikedetect as sd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Load template from a previous run
params = sd.SpikeDetectionParams.default(fs=50000)
params.spike_template = np.load("my_template.npy")

# Process all .mat files in a directory
data_dir = Path("data/experiment_001")
results = {}

for mat_file in sorted(data_dir.glob("*.mat")):
    rec = sd.load_recording(mat_file)
    result = sd.detect_spikes(rec, params)
    results[rec.name] = result
    print(f"{mat_file.name}: {result.n_spikes} spikes")

# Summary
total = sum(r.n_spikes for r in results.values())
print(f"\nTotal: {total} spikes across {len(results)} recordings")
```

---

## Understanding the Output

### SpikeDetectionResult

```python
result = sd.detect_spikes(rec, params)

result.spike_times            # int64 array -- 0-based sample indices
result.spike_times_uncorrected # before inflection point correction
result.spike_times_seconds    # float array -- times in seconds
result.n_spikes               # number of spikes found
result.params                 # the params used for this detection
result.spot_checked           # True if manually reviewed
```

### Summary output

```python
print(result.summary())
# Spike Detection Result
#   Spikes found: 296
#   Time range: 0.162 - 7.882 s
#   Mean ISI: 26.2 ms (range 5.3 - 112.4 ms)
#   Mean firing rate: 38.2 Hz
#   Spot-checked: no
```

### Export to DataFrame

```python
df = result.to_dataframe()
# Columns: spike_index, spike_time_s, spike_index_uncorrected
df.to_csv("spikes.csv", index=False)
```

### Plotting

```python
# Plot voltage trace with spike markers
rec.result = result
fig = rec.plot(show_spikes=True)

import matplotlib.pyplot as plt
plt.show()
```

---

## Tips for Getting Good Results

### Choosing filter settings
- Start with `SpikeDetectionParams.default(fs=your_rate)` -- the defaults work for most recordings
- If your spikes are in a specific frequency band, adjust `hp_cutoff` and `lp_cutoff` accordingly
- `diff_order=1` (first derivative) usually gives the cleanest peaks
- If your spikes are downward-going, set `polarity=-1`

### Getting a good template
- Select 3-5 clear, well-isolated spikes in the `TemplateSelectionGUI`
- Avoid selecting spikes that overlap with other spikes or artifacts
- The template should represent a "typical" spike, not the largest or smallest

### Tuning thresholds
- Use the `ThresholdGUI` scatter plot to see the distribution
- Good spikes cluster in the lower-left (low distance, high amplitude)
- Noise/artifacts cluster in the upper-right
- Set thresholds to separate these clusters

### When to use SpotCheck
- Always spot-check your final results for publication-quality data
- SpotCheck lets you manually accept/reject individual spikes
- Especially important for borderline cases near the threshold boundaries

### Cross-validation with MATLAB
- If you have previous MATLAB results, load the .mat file and compare spike counts
- The Python pipeline produces identical results (verified: 296/296 match on test data)

---

## API Quick Reference

### Top-level functions

```python
import spikedetect as sd

sd.detect_spikes(recording, params)    # Run full pipeline
sd.load_recording("file.mat")         # Load .mat file
sd.load_abf("file.abf")              # Load ABF file
sd.save_result("file.mat", rec, res)  # Save results to .mat
```

### Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `Recording` | `spikedetect.models` | Holds voltage data + metadata |
| `SpikeDetectionParams` | `spikedetect.models` | All detection parameters |
| `SpikeDetectionResult` | `spikedetect.models` | Detection output (spike times) |
| `SpikeDetector` | `spikedetect.pipeline.detect` | Full pipeline orchestrator |
| `SignalFilter` | `spikedetect.pipeline.filtering` | Butterworth bandpass filter |
| `PeakFinder` | `spikedetect.pipeline.peaks` | Peak detection |
| `TemplateMatcher` | `spikedetect.pipeline.template` | DTW template matching |
| `DTW` | `spikedetect.pipeline.dtw` | Dynamic Time Warping distance |
| `InflectionPointDetector` | `spikedetect.pipeline.inflection` | Spike timing correction |
| `SpikeClassifier` | `spikedetect.pipeline.classify` | Quality classification |
| `WaveformProcessor` | `spikedetect.utils` | Smoothing/differentiation |
| `FilterGUI` | `spikedetect.gui` | Filter parameter tuning |
| `TemplateSelectionGUI` | `spikedetect.gui` | Spike template selection |
| `ThresholdGUI` | `spikedetect.gui` | Threshold adjustment |
| `SpotCheckGUI` | `spikedetect.gui` | Manual spike review |
