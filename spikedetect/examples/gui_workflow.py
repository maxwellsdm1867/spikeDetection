#!/usr/bin/env python
"""Complete interactive GUI workflow for spike detection.

Walks through all 4 spikedetect GUIs in order:
  1. FilterGUI             -- tune bandpass filter parameters
  2. TemplateSelectionGUI  -- click peaks to build a spike template
  3. ThresholdGUI          -- adjust DTW distance and amplitude thresholds
  4. SpotCheckGUI          -- review individual spikes (click raster ticks
                              or scatter dots to navigate)

Works with real files (.mat, .abf) or generates synthetic data if no
file is provided.

Usage
-----
    # Synthetic demo (no file needed)
    python examples/gui_workflow.py

    # Load a MATLAB .mat file
    python examples/gui_workflow.py path/to/trial.mat

    # Load an ABF file
    python examples/gui_workflow.py path/to/recording.abf

Requirements
------------
    pip install -e .              # core (synthetic demo)
    pip install -e ".[io]"        # for .abf or HDF5 .mat support
"""

import sys

import numpy as np

import spikedetect as sd
from spikedetect.gui import (
    FilterGUI,
    TemplateSelectionGUI,
    ThresholdGUI,
    SpotCheckGUI,
)
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher


def make_synthetic_recording(
    fs: int = 10000,
    duration: float = 5.0,
    n_spikes: int = 8,
    seed: int = 42,
) -> tuple[sd.Recording, np.ndarray]:
    """Generate a synthetic recording with embedded Gaussian spikes."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration)
    voltage = rng.normal(0, 0.001, n_samples)

    margin = int(0.1 * fs)
    positions = np.linspace(margin, n_samples - margin, n_spikes, dtype=int)

    for pos in positions:
        t_local = np.arange(-25, 26) / fs
        spike = 5.0 * np.exp(-0.5 * (t_local / 0.0003) ** 2)
        start = pos - 25
        end = pos + 26
        if start >= 0 and end <= n_samples:
            voltage[start:end] += spike

    rec = sd.Recording(name="synthetic_demo", voltage=voltage, sample_rate=fs)
    return rec, positions


def main():
    # ---- Load or generate data ------------------------------------------
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
        print(f"Loading {filepath}...")
        if filepath.endswith(".abf"):
            rec = sd.load_abf(filepath)
        else:
            rec = sd.load_recording(filepath)
        true_positions = None
    else:
        print("No file provided -- generating synthetic recording")
        print("  Tip: python examples/gui_workflow.py <file.mat|file.abf>")
        rec, true_positions = make_synthetic_recording()

    fs = rec.sample_rate
    print(f"  {rec.name}: {rec.n_samples} samples, {fs} Hz, {rec.duration:.1f} s")
    print()

    # Reuse existing params if the file had them, otherwise start fresh
    if rec.result is not None and rec.result.params.spike_template is not None:
        print("  Found saved detection parameters -- using as starting point.")
        params = rec.result.params
    else:
        params = sd.SpikeDetectionParams.default(fs=fs)

    # ---- Step 1: FilterGUI ----------------------------------------------
    print("Step 1: FilterGUI -- Tune filter parameters")
    print("  Adjust sliders, then press Enter to accept.")
    filter_gui = FilterGUI(rec.voltage, params)
    params = filter_gui.run()
    print(f"  -> hp={params.hp_cutoff} Hz, lp={params.lp_cutoff} Hz, "
          f"diff={params.diff_order}, pol={params.polarity:+d}")
    print()

    # ---- Step 2: Filter the data ----------------------------------------
    start_point = round(0.01 * fs)
    unfiltered_data = rec.voltage[start_point:]
    filtered_data = SignalFilter.filter_data(
        unfiltered_data,
        fs=fs,
        hp_cutoff=params.hp_cutoff,
        lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order,
        polarity=params.polarity,
    )

    # ---- Step 3: TemplateSelectionGUI -----------------------------------
    print("Step 2: TemplateSelectionGUI -- Click peaks to build a template")
    print("  Click 3-5 clear peaks, then press Enter.")
    template_gui = TemplateSelectionGUI(filtered_data, params)
    template = template_gui.run()

    if template is not None:
        params.spike_template = template
        print(f"  -> Template length: {len(template)} samples")
    elif params.spike_template is not None:
        print("  -> No new selection, keeping existing template.")
    elif true_positions is not None:
        # Fallback for synthetic demo
        stw = params.spike_template_width
        half = stw // 2
        spike_in_trimmed = true_positions[0] - start_point
        params.spike_template = filtered_data[
            spike_in_trimmed - half:spike_in_trimmed + half + 1
        ].copy()
        print(f"  -> Auto-extracted template ({len(params.spike_template)} samples)")
    else:
        print("  ERROR: No template selected and none available. Exiting.")
        sys.exit(1)
    print()

    # ---- Step 4: Initial detection --------------------------------------
    print("Step 3: Running initial spike detection...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 5: ThresholdGUI -------------------------------------------
    print("Step 4: ThresholdGUI -- Adjust distance/amplitude thresholds")
    print("  Click scatter plot to move threshold. 'b' to toggle. Enter to accept.")
    spike_locs = PeakFinder.find_spike_locations(
        filtered_data,
        peak_threshold=params.peak_threshold,
        fs=fs,
        spike_template_width=params.spike_template_width,
    )
    match_result = TemplateMatcher.match(
        spike_locs=spike_locs,
        spike_template=params.spike_template,
        filtered_data=filtered_data,
        unfiltered_data=unfiltered_data,
        spike_template_width=params.spike_template_width,
        fs=fs,
    )
    threshold_gui = ThresholdGUI(match_result, params)
    params = threshold_gui.run()
    print(f"  -> distance_threshold={params.distance_threshold:.2f}, "
          f"amplitude_threshold={params.amplitude_threshold:.3f}")
    print()

    # ---- Step 6: Re-detect with updated thresholds ----------------------
    print("Step 5: Re-running detection with updated thresholds...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 7: SpotCheckGUI -------------------------------------------
    print("Step 6: SpotCheckGUI -- Review spikes one by one")
    print("  y=accept, n=reject, arrows=adjust, click raster ticks to jump,")
    print("  click scatter dots to navigate, Enter=done")
    spotcheck = SpotCheckGUI(rec, result)
    result = spotcheck.run()
    print()

    # ---- Summary --------------------------------------------------------
    print("=" * 50)
    print(result.summary())
    print("=" * 50)


if __name__ == "__main__":
    main()
