# GUI Performance Optimization Design

Future optimization plan for making the spikedetect interactive GUIs more responsive during slider drags, threshold clicks, and spike navigation.

## Problem

The GUIs feel sluggish on large recordings (e.g., 400k samples at 50kHz). Every user interaction triggers:

1. **Full signal recomputation** (filtering 400k samples, peak finding)
2. **Complete matplotlib redraws** (`cla()` clears all artists, then replots from scratch)

## Root Cause Analysis

### FilterGUI (worst offender)

Every slider tick triggers `_apply_filter()` which:

- Designs NEW Butterworth filter coefficients via `butter()` (~5-10ms)
- Runs `lfilter()` on all 400k samples twice (HP + LP) (~20-30ms)
- Optionally differentiates (~5ms)
- Calls `find_spike_locations()` on 400k samples (~10ms)

Then `_update_plots()` calls `cla()` on BOTH axes and replots all 400k points.

The unfiltered trace in the top panel **never changes** but is replotted every time.

Total: ~60-80ms per slider tick, and sliders generate many events during a drag.

### ThresholdGUI

Every click calls `_update_panels()` which:

- Calls `cla()` on ALL 6 axes (scatter + 4 waveform panels + mean panel)
- Replots all scatter dots, waveform overlays, recomputes mean + 2nd derivative
- All of this just to move a threshold line

### SpotCheckGUI

Every y/n/arrow/tab calls `_show_current_spike()` which:

- Calls `cla()` on spike context and filtered context axes
- Replots voltage, spike highlight, mean waveform overlay, template overlay

### TemplateSelectionGUI

Already uses incremental updates (adds plot elements on each click). No significant issues.

## What Can vs Can't Be Precomputed

Not everything can be cached -- the key constraint is that users make decisions interactively:

### FilterGUI

- `filter_data()` **MUST rerun** on every slider change (depends on user-chosen HP, LP, diff_order, polarity)
- `find_spike_locations()` **MUST rerun** (depends on new filtered data + peak_threshold)
- **Can precompute**: unfiltered trace (never changes), matplotlib artist objects (Line2D refs)
- **Can cache**: `butter()` coefficients per (cutoff, fs) pair -- avoids redundant filter design
- **Big win**: debounce -- don't recompute 10x during a slider drag, only once when user pauses

### ThresholdGUI

- DTW distances + amplitudes are **already precomputed** in `match_result` -- no recomputation needed
- `classify_spikes()` is cheap (~1ms) but waveform panel redraws are expensive
- **Can precompute**: scatter dots, threshold lines (create once as persistent artists)
- **Big win**: don't `cla()` + replot scatter on every click -- just move the threshold line

### SpotCheckGUI

- `_setup()` **already precomputes** everything at load (filter + template matching)
- Per-keystroke only redraws context window around current spike
- **Can precompute**: full-trace plots (drawn once), Line2D refs for context panels
- **Big win**: don't `cla()` context panels -- just `set_data()` on existing lines

## Proposed Approach: Config-Based Rendering Modes

Add a `GUIConfig` dataclass that controls rendering behavior. Two presets:

- **`"fast"` (default)** -- debounced sliders, retained plot objects, conservative downsampling, cached filter coefficients
- **`"full"`** -- current behavior unchanged (full redraws, no downsampling, no debounce)

```python
from spikedetect.gui import GUIConfig

# Fast mode (default -- you don't need to do anything)
filter_gui = FilterGUI(voltage, params)

# Full mode (exact current behavior)
config = GUIConfig.full()
filter_gui = FilterGUI(voltage, params, config=config)

# Custom
config = GUIConfig(debounce_ms=100, downsample_max_points=8000)
filter_gui = FilterGUI(voltage, params, config=config)
```

### GUIConfig Fields

| Field | Default (fast) | Full mode | What it controls |
|-------|---------------|-----------|-----------------|
| `debounce_ms` | 50 | 0 | Delay before recomputing after slider stops moving |
| `downsample_max_points` | 20000 | 0 | Max points for display traces (0 = all points) |
| `retain_artists` | True | False | Reuse Line2D objects vs `cla()` + recreate |
| `cache_classification` | True | False | Skip redundant `classify_spikes()` calls |

## Optimization Details

### 1. Debounce Slider Events (FilterGUI)

Replace immediate `_apply_filter()` + `_update_plots()` with a timer-based debounce using `fig.canvas.new_timer()`. The slider callback updates `self.params` immediately (so the param values are always current) but delays the expensive recomputation until the user pauses.

Discrete events (polarity toggle, diff radio) still fire immediately since those are single clicks.

**Impact**: Eliminates 95%+ of redundant recomputations during slider drags.

### 2. Cache Butterworth Coefficients (filtering.py)

Add `@lru_cache(maxsize=32)` wrapper around `scipy.signal.butter()`. This is always-on (no config toggle needed) -- there is zero downside to caching deterministic filter coefficients.

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def _cached_butter(order: int, wn: float, btype: str):
    return butter(order, wn, btype=btype)
```

With slider `valstep=0.5`, the same Hz values repeat often during interaction.

**Impact**: ~5-10ms saved per filter call.

### 3. Retain Plot Objects (all GUIs)

The core pattern: create persistent `Line2D` references in `_build_figure()`, then update data via `set_ydata()` / `set_data()` instead of `cla()` + replot.

**FilterGUI example**:

```python
# In _build_figure() -- create once:
self._line_unfilt, = self._ax_unfilt.plot(x, self._unfiltered, ...)
self._line_filt, = self._ax_filt.plot(x, np.zeros(n), "k", linewidth=0.5)
self._line_peaks, = self._ax_filt.plot([], [], "ro", markersize=4)

# In _update_plots() -- update data only:
self._line_filt.set_ydata(filt)
self._line_peaks.set_data(self._locs, filt[self._locs])
# Never touch _line_unfilt -- it doesn't change
```

The unfiltered trace (top panel) is plotted once and never redrawn.

**Impact**: ~15-20ms saved per update by avoiding matplotlib artist creation/destruction.

### 4. Display Downsampling

Add a `downsample_minmax()` helper that uses min-max decimation: for every block of N samples, keep only the min and max values. This preserves the exact amplitude envelope -- no spikes or features are visually lost.

Default: 400k samples to ~20k points (~16 points per pixel on a 1200px display). All pipeline computations still use full-resolution data -- only display rendering is downsampled.

**Impact**: ~10-15ms render savings per update.

### 5. Cache Classification (ThresholdGUI)

Store previous `(distance_threshold, amplitude_threshold)` and the resulting classification masks. If thresholds haven't changed, skip `classify_spikes()` and waveform panel redraws entirely -- only update scatter threshold line positions.

**Impact**: Skip unnecessary waveform redraws when clicking the same scatter region.

## Files to Modify

| File | Changes |
|------|---------|
| `gui/_widgets.py` | Add `GUIConfig` dataclass, `downsample_minmax()` helper |
| `gui/filter_gui.py` | Accept config, conditionally debounce/retain/downsample |
| `gui/threshold_gui.py` | Accept config, conditionally retain scatter artists |
| `gui/spotcheck_gui.py` | Accept config, conditionally retain artists + downsample |
| `gui/__init__.py` | Export `GUIConfig` |
| `gui/workflow.py` | Pass config through to all GUIs |
| `pipeline/filtering.py` | Add `_cached_butter()` with `lru_cache` (always-on) |

## Expected Impact

| Optimization | Savings per interaction | Applies to |
|---|---|---|
| Debounce sliders | Eliminates 95%+ redundant recomputations during drag | FilterGUI |
| Retain plot objects | ~15-20ms (no `cla()` overhead) | All GUIs |
| Cache butter coefficients | ~5-10ms per filter call | FilterGUI |
| Display downsampling | ~10-15ms render time | FilterGUI, SpotCheckGUI |
| Cache classification | Skip waveform redraws when thresholds unchanged | ThresholdGUI |

## Testing Considerations

- All 135 existing tests must pass (`python -m pytest tests/ -v`)
- Tests use Agg backend and check attributes/state, not rendered plot contents
- `test_slider_change_updates_params`: params set before debounce timer, still passes
- `test_scatter_plot_has_dots`: retained artists satisfy `gui._scat_in is not None`
- Cross-validation unaffected (pipeline math unchanged)
- Add `TestGUIConfig` verifying fast/full presets and custom config
