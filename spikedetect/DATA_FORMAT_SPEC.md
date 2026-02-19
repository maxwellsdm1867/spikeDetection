# Data Format Specification

This document defines the data structures and file formats used by the `spikedetect` package. It is intended as a contract: any acquisition frontend or data translator that produces data conforming to this spec can be used with the spike detection pipeline.

---

## Core Data Model

The pipeline operates on two objects: a **Recording** (input) and **SpikeDetectionParams** (configuration). Detection produces a **SpikeDetectionResult** (output).

### Recording

A single electrophysiology sweep/trial.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Human-readable identifier (e.g., filename, trial ID) |
| `voltage` | 1-D float64 array | yes | Membrane voltage trace in **volts** (not mV or uV). Must be non-empty. |
| `sample_rate` | float > 0 | yes | Sampling rate in **Hz** (e.g., 10000 for 10 kHz, 50000 for 50 kHz) |
| `current` | 1-D float64 array or null | no | Injected current trace in **amperes**. Same length as `voltage` if present. |
| `metadata` | dict | no | Arbitrary key-value pairs (protocol name, date, experimenter, etc.) |
| `result` | SpikeDetectionResult or null | no | Previously computed detection results, if any |

**Voltage units matter.** The pipeline's default thresholds assume voltage is in volts (SI units). If your acquisition system records in millivolts or microvolts, scale before passing to the pipeline:

```python
# Example: convert millivolts to volts
recording = Recording(
    name="trial_001",
    voltage=raw_mv * 1e-3,   # mV -> V
    sample_rate=10000,
)
```

### SpikeDetectionParams

Configuration for the detection pipeline. All fields have defaults; only `fs` is strictly required.

| Field | Type | Default | Description |
|---|---|---|---|
| `fs` | float | (required) | Sample rate in Hz. Must match `Recording.sample_rate`. |
| `spike_template_width` | int | `round(0.005 * fs) + 1` | Template window width in samples. Auto-computed from `fs` if 0. |
| `hp_cutoff` | float | 200.0 | High-pass filter cutoff in Hz. Must be < `fs/2`. |
| `lp_cutoff` | float | 800.0 | Low-pass filter cutoff in Hz. Must be < `fs/2`. |
| `diff_order` | int | 1 | Differentiation order after bandpass: 0 (none), 1 (first derivative), or 2 (second). |
| `peak_threshold` | float | 5.0 | Added to mean(filtered_data) to get the minimum peak height. |
| `distance_threshold` | float | 15.0 | Maximum DTW distance for a candidate to be accepted as a spike. |
| `amplitude_threshold` | float | 0.2 | Minimum amplitude projection for a candidate to be accepted. |
| `spike_template` | 1-D float64 array or null | null | Reference spike waveform for DTW matching. **Required for detection.** |
| `polarity` | int | 1 | +1 for upward spikes, -1 for downward spikes. |
| `likely_inflection_point_peak` | int or null | null | Pre-computed inflection point index (0-based). Computed automatically if null. |
| `last_filename` | string | "" | Last processed filename (for parameter persistence). |

### SpikeDetectionResult

Output of the detection pipeline.

| Field | Type | Description |
|---|---|---|
| `spike_times` | 1-D int64 array | Corrected spike times as **0-based sample indices** into the original `voltage` array |
| `spike_times_uncorrected` | 1-D int64 array | Uncorrected spike times (peak locations before inflection point correction) |
| `params` | SpikeDetectionParams | The parameters used for this detection |
| `spot_checked` | bool | Whether the results have been manually reviewed |

**Index convention:** All spike times are **0-based sample indices** into the original `Recording.voltage` array. To convert to seconds: `spike_times / sample_rate`.

---

## File Formats

The package supports three file formats for input and one native format for round-trip storage.

### 1. MATLAB .mat Files (primary input format)

Loaded via `spikedetect.io.load_recording(path)`. Supports both MATLAB v5/v7 (via scipy) and v7.3 HDF5 (via h5py).

#### Required fields

| .mat variable | Type | Maps to |
|---|---|---|
| `voltage_1` | numeric vector | `Recording.voltage` (flattened to 1-D float64) |
| `params.sampratein` | scalar | `Recording.sample_rate` |

#### Optional fields

| .mat variable | Type | Maps to |
|---|---|---|
| `name` | string (or char array) | `Recording.name` |
| `current_2` | numeric vector | `Recording.current` |
| `spikes` | numeric vector | `SpikeDetectionResult.spike_times` (loaded as int64) |
| `spikes_uncorrected` | numeric vector | `SpikeDetectionResult.spike_times_uncorrected` |
| `spikeSpotChecked` | scalar (0 or 1) | `SpikeDetectionResult.spot_checked` |
| `spikeDetectionParams` | struct (see below) | `SpikeDetectionParams` |

#### spikeDetectionParams struct

If present in the .mat file, this struct is loaded into `SpikeDetectionParams`. The field names use the original MATLAB naming convention:

| Struct field | Python param | Type |
|---|---|---|
| `fs` | `fs` | scalar float |
| `spikeTemplateWidth` | `spike_template_width` | scalar int |
| `hp_cutoff` | `hp_cutoff` | scalar float |
| `lp_cutoff` | `lp_cutoff` | scalar float |
| `diff` | `diff_order` | scalar int (0, 1, or 2) |
| `peak_threshold` | `peak_threshold` | scalar float |
| `Distance_threshold` | `distance_threshold` | scalar float |
| `Amplitude_threshold` | `amplitude_threshold` | scalar float |
| `spikeTemplate` | `spike_template` | 1-D numeric vector |
| `polarity` | `polarity` | scalar int (-1 or 1) |
| `likelyiflpntpeak` | `likely_inflection_point_peak` | scalar int |
| `lastfilename` | `last_filename` | string |

**Minimal .mat file example** (MATLAB code to create one):

```matlab
voltage_1 = randn(1, 100000) * 1e-3;  % 1-D voltage in volts
params.sampratein = 10000;             % 10 kHz
save('my_trial.mat', 'voltage_1', 'params', '-v7');
```

**Minimal .mat file example** (Python code to create one):

```python
import numpy as np
import scipy.io

scipy.io.savemat('my_trial.mat', {
    'voltage_1': np.random.randn(100000) * 1e-3,  # volts
    'params': {'sampratein': np.array([10000.0])},
}, do_compression=True)
```

### 2. ABF Files (Axon Binary Format)

Loaded via `spikedetect.io.load_abf(path)`. Requires `pyabf >= 2.3`.

| ABF field | Maps to |
|---|---|
| Channel 0 sweep data | `Recording.voltage` |
| Channel 1 sweep data (if present) | `Recording.current` |
| `abf.dataRate` | `Recording.sample_rate` |
| File path | `Recording.name` |

ABF files cannot store detection params or results. After detection, save results using the native HDF5 format or MATLAB .mat format.

**Note:** ABF files from pClamp/Clampex typically store voltage in millivolts or picoamperes. Check the `abf.adcUnits` property and scale to SI units (volts, amperes) if needed.

### 3. Native HDF5 Format (.h5)

Round-trip format for saving and loading recordings with results. Uses `h5py`.

- **Save:** `spikedetect.io.save_native(path, recording)`
- **Load:** `spikedetect.io.load_native(path)`

#### HDF5 structure

```
/
├── voltage          (dataset, float64, gzip-compressed)
├── current          (dataset, float64, gzip-compressed, optional)
├── @name            (attribute, string)
├── @sample_rate     (attribute, float)
├── @metadata        (attribute, JSON string, optional)
└── result/          (group, optional)
    ├── spike_times            (dataset, int64)
    ├── spike_times_uncorrected (dataset, int64)
    ├── @spot_checked          (attribute, bool)
    └── @params                (attribute, JSON string → SpikeDetectionParams.to_dict())
```

### 4. MATLAB .mat Output Format

Written via `spikedetect.io.save_result(path, recording)`. Produces MATLAB v7.3 (HDF5) files with the same field layout as input .mat files (see section 1), so the files can be read back by both MATLAB and Python.

### 5. JSON Parameter Files

Stored in `~/.spikedetect/` for parameter persistence across sessions.

- **Save:** `spikedetect.io.config.save_params(params, input_field='voltage_1')`
- **Load:** `spikedetect.io.config.load_params(input_field='voltage_1', fs=10000)`
- **File naming:** `Spike_params_{input_field}_fs{fs}.json`

The JSON schema matches `SpikeDetectionParams.to_dict()`:

```json
{
  "fs": 50000.0,
  "spike_template_width": 251,
  "hp_cutoff": 834.63,
  "lp_cutoff": 160.24,
  "diff_order": 1,
  "peak_threshold": 0.0001,
  "distance_threshold": 9.5946,
  "amplitude_threshold": -0.0764,
  "polarity": -1,
  "last_filename": "trial_001.mat",
  "spike_template": [0.001, 0.003, ...],
  "likely_inflection_point_peak": 174
}
```

All fields except `fs` are optional when deserializing (defaults are used).

---

## Writing a Data Translator

To integrate a new acquisition system with `spikedetect`, you need to produce a `Recording` object. Here is the minimal contract:

### Option A: Create a Recording object directly (recommended)

```python
import numpy as np
from spikedetect import Recording

recording = Recording(
    name="my_trial_001",
    voltage=voltage_array,     # 1-D numpy float64 array, in VOLTS
    sample_rate=10000.0,       # Hz
    current=current_array,     # optional, same length as voltage, in AMPERES
)
```

### Option B: Write a .mat file that load_recording() can read

Your translator must produce a .mat file (v5, v7, or v7.3) with at minimum:

```
voltage_1    : 1-D or 2-D numeric array (will be flattened)
params       : struct with field 'sampratein' (scalar, Hz)
```

Everything else is optional. See section 1 for the full field list.

### Option C: Write a native .h5 file

```python
import h5py
import numpy as np

with h5py.File('my_trial.h5', 'w') as f:
    f.attrs['name'] = 'my_trial_001'
    f.attrs['sample_rate'] = 10000.0
    f.create_dataset('voltage', data=voltage_array, compression='gzip')
    # optional:
    f.create_dataset('current', data=current_array, compression='gzip')
```

### Checklist for translators

1. **Voltage in volts.** If your system records mV, uV, or ADC counts, convert to volts (SI).
2. **1-D array.** Flatten any multi-dimensional voltage/current arrays to 1-D.
3. **float64 dtype.** Integer or float32 data will be cast, but float64 avoids precision loss.
4. **Positive sample rate in Hz.** Not kHz, not samples/ms.
5. **Non-empty voltage.** At least 1 sample. In practice, the pipeline needs enough data for the filter transient (first 1% is trimmed) and the template width.
6. **Consistent lengths.** If `current` is provided, it must have the same number of samples as `voltage`.

### Minimum viable recording for detection

To run the full pipeline, you need:
- A `Recording` with voltage and sample_rate
- A `SpikeDetectionParams` with `spike_template` set (a 1-D waveform of a representative spike)

```python
import spikedetect as sd

recording = Recording(name="test", voltage=my_voltage, sample_rate=50000)
params = sd.SpikeDetectionParams.default(fs=50000)
params.spike_template = my_template  # 1-D array, typically ~5ms wide
result = sd.detect_spikes(recording, params)
```

---

## Appendix: Type Reference

### numpy dtype requirements

| Field | dtype | Notes |
|---|---|---|
| `voltage` | float64 | Auto-cast on construction |
| `current` | float64 | Auto-cast on construction |
| `spike_template` | float64 | Auto-cast on construction |
| `spike_times` | int64 | 0-based sample indices |
| `spike_times_uncorrected` | int64 | 0-based sample indices |

### Value constraints

| Field | Constraint |
|---|---|
| `sample_rate` / `fs` | > 0 |
| `hp_cutoff` | > 0, < fs/2 |
| `lp_cutoff` | > 0, < fs/2 |
| `diff_order` | 0, 1, or 2 |
| `polarity` | -1 or +1 |
| `voltage` | non-empty |
| `spike_template` | 1-D if provided |

### Serialization round-trip

```
SpikeDetectionParams  <-->  dict  <-->  JSON string
    .to_dict()         json.dumps()
    .from_dict()       json.loads()
```

This serialization is used by:
- JSON config files (`~/.spikedetect/`)
- Native HDF5 format (params stored as JSON attribute)
- Any custom integration that needs to persist/transmit parameters
