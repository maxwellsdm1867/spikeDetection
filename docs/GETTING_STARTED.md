# Getting Started with spikedetect

This guide walks you through installing `spikedetect` and running your first spike detection in under 5 minutes.

## Prerequisites

- **Python 3.9 or later** (check with `python --version`)
- **pip** package manager
- A recording file (`.mat` or `.abf`)

If you don't have Python yet, install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) -- both include Python, pip, and scientific libraries.

## Step 1: Install

```bash
# Clone the repository
git clone https://github.com/tony-azevedo/spikeDetection.git
cd spikeDetection/spikedetect

# Install the package (editable mode so you can modify if needed)
pip install -e .
```

To verify the install worked:

```bash
python -c "import spikedetect; print(spikedetect.__version__)"
# Should print: 0.1.0
```

### Optional extras

```bash
# Faster DTW computation (numba JIT acceleration)
pip install -e ".[fast]"

# ABF and HDF5 file support
pip install -e ".[io]"

# Everything (fast + io + dev tools)
pip install -e ".[all]"
```

## Step 2: Load a recording

### From a MATLAB .mat file (FlyAnalysis trial struct)

```python
import spikedetect as sd

rec = sd.load_recording("path/to/trial.mat")
print(f"Loaded: {rec.name}")
print(f"  {rec.n_samples} samples, {rec.sample_rate} Hz, {rec.duration:.2f} s")
```

If the .mat file already has spike detection parameters and results from a previous MATLAB run, they are automatically loaded into `rec.result`.

### From an ABF file

```python
rec = sd.load_abf("path/to/recording.abf")
```

### From a raw numpy array

```python
import numpy as np

voltage = np.load("my_voltage.npy")  # or however you have your data
rec = sd.Recording(
    name="experiment_001",
    voltage=voltage,
    sample_rate=50000,  # Hz
)
```

## Step 3: Set up parameters

```python
# Create default parameters for your sample rate
params = sd.SpikeDetectionParams.default(fs=rec.sample_rate)
```

This gives you sensible starting values. The key parameters are:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `hp_cutoff` | 200 Hz | High-pass filter cutoff |
| `lp_cutoff` | 800 Hz | Low-pass filter cutoff |
| `diff_order` | 1 | Differentiation order (0, 1, or 2) |
| `polarity` | 1 | Flip signal (+1 or -1) |
| `peak_threshold` | 5.0 | Minimum peak height for candidates |
| `distance_threshold` | 15.0 | Maximum DTW distance to accept a spike |
| `amplitude_threshold` | 0.2 | Minimum spike amplitude |

You'll need a **spike template** before running detection. Either:
- Use the interactive GUI (Step 4a), or
- Provide one from a previous run (Step 4b)

## Step 4a: Interactive workflow (recommended for new data)

```python
from spikedetect.gui import FilterGUI, TemplateSelectionGUI, ThresholdGUI, SpotCheckGUI

# 1. Tune filter settings with live sliders
filter_gui = FilterGUI(rec.voltage, params)
params = filter_gui.run()

# 2. Click on peaks to select seed spikes for the template
template_gui = TemplateSelectionGUI(filtered_data, params)
params.spike_template = template_gui.run()

# 3. Run detection
result = sd.detect_spikes(rec, params)

# 4. Fine-tune distance/amplitude thresholds
threshold_gui = ThresholdGUI(match_result, params)
params = threshold_gui.run()

# 5. Review individual spikes (y/n/arrow keys)
spotcheck = SpotCheckGUI(rec, result)
result = spotcheck.run()
```

In Jupyter notebooks, use `%matplotlib widget` at the top of your notebook (requires `pip install ipympl`).

## Step 4b: Batch detection (when you already have a template)

If you have parameters from a previous run (e.g., loaded from the .mat file):

```python
# Use the params that were loaded with the recording
params_from_file = rec.result.params  # if the .mat had previous detection results

result = sd.detect_spikes(rec, params_from_file)
```

Or set the template manually:

```python
params.spike_template = my_template_array  # 1-D numpy array
result = sd.detect_spikes(rec, params)
```

## Step 5: Inspect results

```python
# Quick summary
print(result.summary())
# Output:
#   Spike Detection Result
#     Spikes found: 296
#     Time range: 0.162 - 7.882 s
#     Mean ISI: 26.2 ms (range 5.3 - 112.4 ms)
#     Mean firing rate: 38.2 Hz
#     Spot-checked: no

# Spike times as sample indices (0-based)
print(result.spike_times[:10])

# Spike times in seconds
print(result.spike_times_seconds[:10])

# Number of spikes
print(result.n_spikes)

# Plot the recording with spike markers
rec.result = result
fig = rec.plot()
```

### Export to pandas DataFrame

```python
df = result.to_dataframe()
print(df.head())
#    spike_index  spike_time_s  spike_index_uncorrected
# 0         8127      0.162540                     8130
# 1        10794      0.215880                    10800
# ...
```

### Export to CSV

```python
df = result.to_dataframe()
df.to_csv("spike_times.csv", index=False)
```

## Step 6: Save your work

### Save parameters for reuse

```python
from spikedetect.io.config import save_params, load_params

# Save to ~/.spikedetect/experiment_001.json
save_params(params, "experiment_001")

# Load back later
params = load_params("experiment_001")
```

### Save results back to .mat

```python
from spikedetect.io.mat import save_result

save_result("trial_with_spikes.mat", rec, result)
```

### Save in native HDF5 format

```python
from spikedetect.io.native import save_native, load_native

save_native("output.h5", rec)
rec_loaded = load_native("output.h5")
```

## Enable logging

To see what the pipeline is doing at each step:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now detect_spikes will print progress:
# INFO:spikedetect.pipeline.detect:Starting spike detection on 'trial_001' (8.0 s, 50000 Hz)
# INFO:spikedetect.pipeline.detect:Filtering: hp=800 Hz, lp=160 Hz, diff_order=1, polarity=-1
# INFO:spikedetect.pipeline.detect:Found 342 candidate peaks
# INFO:spikedetect.pipeline.detect:Accepted 296 / 342 candidates (distance < 9.6, amplitude > -0.080)
# INFO:spikedetect.pipeline.detect:Detection complete: 296 spikes found in 'trial_001'
```

## Troubleshooting

### "No spike template provided"
You need to set `params.spike_template` before calling `detect_spikes()`. Use the interactive `TemplateSelectionGUI` or provide a 1-D numpy array from a previous run.

### "High-pass cutoff must be below the Nyquist frequency"
Your filter cutoff is too high for your sample rate. The cutoff must be less than `sample_rate / 2`. Use `SpikeDetectionParams.default(fs=your_rate)` to get auto-scaled defaults.

### Very few or zero spikes detected
- Try lowering `params.peak_threshold` to find more candidates
- Try increasing `params.distance_threshold` to accept more candidates
- Try lowering `params.amplitude_threshold`
- Try `params.polarity = -1` if your spikes go downward
- Use the `FilterGUI` to visually check that filtering reveals spikes

### Too many false positives
- Lower `params.distance_threshold` (stricter template match)
- Raise `params.amplitude_threshold`
- Use `SpotCheckGUI` to manually review and reject bad detections

### numba/numpy version warning
If you see `_ARRAY_API not found` warnings, your numba and numpy versions are incompatible. This is harmless -- the pipeline falls back to pure numpy. To silence it, either:
- Uninstall numba: `pip uninstall numba`
- Or upgrade numba: `pip install numba --upgrade`

## Next steps

- Read the [User Guide](USER_GUIDE.md) for detailed explanations of every parameter and pipeline stage
- See [MIGRATION_GUIDE.md](../spikedetect/MIGRATION_GUIDE.md) if you're coming from the MATLAB version
- Run `python -m pytest tests/ -v` to verify your installation
