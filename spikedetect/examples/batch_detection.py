#!/usr/bin/env python
"""Non-interactive batch spike detection with known parameters.

Python equivalent of the MATLAB scripts in ``matlab_reference/scripts/``:
  - Script_SpikeSorting_forSweta.m
  - Script_SpikeSorting_forSydney.m
  - Script_SpikeSorting_forLylah.m
  - Script_detectEMGSpikesForTrial.m

Shows how to:
  1. Load data from an ABF or MAT file
  2. Set up params with a known template (from a prior interactive run)
  3. Optionally chunk long recordings into segments
  4. Run non-interactive detection in a loop
  5. Detect EMG spikes on a current channel
  6. Apply a low-pass pre-filter before detection

Usage
-----
    # Detect spikes on a single file
    python examples/batch_detection.py path/to/recording.abf

    # Or just run the synthetic demo (no file needed)
    python examples/batch_detection.py

Requirements
------------
    pip install -e .              # core only
    pip install -e ".[io]"        # for ABF / HDF5 .mat file support
"""

import sys

import numpy as np

import spikedetect as sd


# -----------------------------------------------------------------------
# Example 1: Non-interactive detection with known parameters
# -----------------------------------------------------------------------

def detect_with_known_params(voltage, fs):
    """Run detection on raw voltage using pre-set parameters.

    This mirrors the MATLAB pattern of setting vars_initial with a saved
    template and calling spikeDetection_forABF(data, vars_initial).

    Parameters
    ----------
    voltage : np.ndarray
        Raw 1-D voltage trace.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sd.SpikeDetectionResult
    """
    # Create a Recording from raw voltage (like MATLAB: data = data(:,1))
    rec = sd.Recording(name="my_recording", voltage=voltage, sample_rate=fs)

    # Set up params with known values from a previous interactive session.
    # This is equivalent to the MATLAB vars_initial struct.
    params = sd.SpikeDetectionParams(
        fs=fs,
        hp_cutoff=400.0,            # MATLAB: vars_initial.hp_cutoff = 400
        lp_cutoff=209.0,            # MATLAB: vars_initial.lp_cutoff = 209.2370
        diff_order=1,               # MATLAB: vars_initial.diff = 1
        polarity=-1,                # MATLAB: vars_initial.polarity = -1
        peak_threshold=3.77e-4,     # MATLAB: vars_initial.peak_threshold
        distance_threshold=11.04,   # MATLAB: vars_initial.Distance_threshold
        amplitude_threshold=0.71,   # MATLAB: vars_initial.Amplitude_threshold
    )

    # Paste in a template from a previous run.
    # In MATLAB this was: vars_initial.spikeTemplate = [4.37e-07, ...];
    # You get this from an interactive session (TemplateSelectionGUI) or
    # from params saved with sd.SpikeDetectionParams.to_dict().
    #
    # For this demo we'll generate one from the data:
    template = _extract_template_from_data(rec, params)
    params.spike_template = template

    # Run non-interactive detection
    result = sd.detect_spikes(rec, params)
    return result


# -----------------------------------------------------------------------
# Example 2: Chunk a long recording and batch-process
# -----------------------------------------------------------------------

def batch_detect_chunks(voltage, fs, chunk_duration=20.0):
    """Split a long recording into chunks and detect spikes in each.

    This mirrors the MATLAB pattern:
        twntysec = 20*vars_initial.fs;
        chcnkdata = reshape(data, twntysec, []);
        for c = 1:size(chcnkdata,2)
            [trial, vars] = spikeDetection_forABF(chcnkdata(:,c), vars_nice);
        end

    Parameters
    ----------
    voltage : np.ndarray
        Raw 1-D voltage trace (can be very long).
    fs : float
        Sampling rate in Hz.
    chunk_duration : float
        Duration of each chunk in seconds (default 20).

    Returns
    -------
    list of sd.SpikeDetectionResult
        One result per chunk, with spike times relative to the chunk start.
    """
    chunk_samples = int(chunk_duration * fs)
    n_chunks = len(voltage) // chunk_samples

    # First, run an interactive session on one chunk to get good params.
    # Here we simulate that with a template extracted from chunk 0.
    first_chunk = voltage[:chunk_samples]
    rec0 = sd.Recording(name="chunk_0", voltage=first_chunk, sample_rate=fs)
    params = sd.SpikeDetectionParams(
        fs=fs, hp_cutoff=400.0, lp_cutoff=209.0,
        diff_order=1, polarity=-1,
        peak_threshold=3.77e-4,
        distance_threshold=11.04,
        amplitude_threshold=0.71,
    )
    params.spike_template = _extract_template_from_data(rec0, params)

    # Now batch-process all chunks with the same params
    results = []
    for c in range(n_chunks):
        start = c * chunk_samples
        end = start + chunk_samples
        chunk = voltage[start:end]

        rec = sd.Recording(
            name=f"chunk_{c}",
            voltage=chunk,
            sample_rate=fs,
        )
        result = sd.detect_spikes(rec, params)
        results.append(result)
        print(f"  Chunk {c + 1}/{n_chunks}: {result.n_spikes} spikes")

    return results


# -----------------------------------------------------------------------
# Example 3: EMG spike detection on a current channel
# -----------------------------------------------------------------------

def detect_emg_spikes(recording):
    """Detect EMG spikes on the current channel of a recording.

    This mirrors the MATLAB script:
        trial.current_2_flipped = sgn * trial.current_2;
        [trial, spikevars] = spikeDetection(trial, 'current_2_flipped', ...);

    Parameters
    ----------
    recording : sd.Recording
        A recording that has a current channel (recording.current).

    Returns
    -------
    sd.SpikeDetectionResult
    """
    if recording.current is None:
        raise ValueError("Recording has no current channel for EMG detection")

    # Create a new Recording using the current channel as "voltage"
    # with optional polarity flip (sgn = 1 or -1)
    sgn = 1
    emg_rec = sd.Recording(
        name=f"{recording.name}_EMG",
        voltage=sgn * recording.current,
        sample_rate=recording.sample_rate,
    )

    params = sd.SpikeDetectionParams.default(fs=recording.sample_rate)
    params.spike_template = _extract_template_from_data(emg_rec, params)

    result = sd.detect_spikes(emg_rec, params)
    return result


# -----------------------------------------------------------------------
# Example 4: Using the pre-filter
# -----------------------------------------------------------------------

def detect_with_pre_filter(voltage, fs):
    """Run detection with a low-pass pre-filter on the raw voltage.

    This mirrors the MATLAB pattern where spikeDetection.m optionally
    calls lowPassFilterMembraneVoltage.m before the detection pipeline.

    Parameters
    ----------
    voltage : np.ndarray
        Raw 1-D voltage trace.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sd.SpikeDetectionResult
    """
    rec = sd.Recording(name="pre_filtered", voltage=voltage, sample_rate=fs)
    params = sd.SpikeDetectionParams(
        fs=fs, hp_cutoff=400.0, lp_cutoff=209.0,
        diff_order=1, polarity=-1,
        peak_threshold=3.77e-4,
        distance_threshold=11.04,
        amplitude_threshold=0.71,
    )
    params.spike_template = _extract_template_from_data(rec, params)

    # Apply a 3 kHz low-pass pre-filter before the detection pipeline.
    # This cleans up high-frequency noise in the raw voltage before
    # the bandpass filter + differentiation stages.
    result = sd.detect_spikes(rec, params, pre_filter_cutoff=3000.0)
    return result


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def _extract_template_from_data(recording, params):
    """Extract a spike template from the first detected peak.

    In real usage you'd get this from TemplateSelectionGUI or a saved
    params dict from a previous session. This helper exists only so
    the demo script can run end-to-end without user interaction.
    """
    from spikedetect.pipeline.filtering import SignalFilter
    from spikedetect.pipeline.peaks import PeakFinder

    start = round(0.01 * params.fs)
    data = recording.voltage[start:]
    filtered = SignalFilter.filter_data(
        data, fs=params.fs,
        hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order, polarity=params.polarity,
    )
    locs = PeakFinder.find_spike_locations(
        filtered, peak_threshold=params.peak_threshold,
        fs=params.fs, spike_template_width=params.spike_template_width,
    )
    if len(locs) == 0:
        raise RuntimeError("No peaks found -- cannot auto-extract template")

    stw = params.spike_template_width
    half = stw // 2
    loc = locs[0]
    return filtered[loc - half:loc + half + 1].copy()


# -----------------------------------------------------------------------
# Main: run synthetic demo or process a real file
# -----------------------------------------------------------------------

def _make_synthetic(fs=10000, duration=60.0, n_spikes=100, seed=42):
    """Generate a long synthetic recording for the demo."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    voltage = rng.normal(0, 0.001, n)

    margin = int(0.5 * fs)
    positions = np.linspace(margin, n - margin, n_spikes, dtype=int)
    for pos in positions:
        t_local = np.arange(-25, 26) / fs
        spike = 5.0 * np.exp(-0.5 * (t_local / 0.0003) ** 2)
        voltage[pos - 25:pos + 26] += spike

    return voltage, fs, positions


def main():
    if len(sys.argv) >= 2:
        # Load a real file
        filepath = sys.argv[1]
        print(f"Loading {filepath}...")
        if filepath.endswith(".abf"):
            rec = sd.load_abf(filepath)
        else:
            rec = sd.load_recording(filepath)
        voltage = rec.voltage
        fs = rec.sample_rate
        print(f"  {len(voltage)} samples, {fs} Hz, {len(voltage)/fs:.1f} s")
    else:
        # Synthetic demo
        print("No file provided -- running synthetic demo")
        print("  Usage: python examples/batch_detection.py <file.abf|file.mat>")
        print()
        voltage, fs, true_pos = _make_synthetic()
        print(f"  Generated {len(voltage)/fs:.0f}s recording with "
              f"{len(true_pos)} embedded spikes at {fs} Hz")

    print()

    # --- Example 1: Single detection with known params ---
    print("Example 1: Non-interactive detection with known params")
    result = detect_with_known_params(voltage, fs)
    print(f"  -> {result.n_spikes} spikes detected")
    print()

    # --- Example 2: Batch chunk processing ---
    print("Example 2: Batch processing in 20s chunks")
    results = batch_detect_chunks(voltage, fs, chunk_duration=20.0)
    total = sum(r.n_spikes for r in results)
    print(f"  -> {total} spikes total across {len(results)} chunks")
    print()

    # --- Example 4: With pre-filter ---
    print("Example 3: Detection with 3 kHz pre-filter")
    result = detect_with_pre_filter(voltage, fs)
    print(f"  -> {result.n_spikes} spikes detected")
    print()

    print("Done!")


if __name__ == "__main__":
    main()
