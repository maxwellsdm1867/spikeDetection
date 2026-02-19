#!/usr/bin/env python
"""Interactive GUI workflow loading data from a .mat or .abf file.

This script performs the same 4-GUI workflow as gui_workflow.py but
loads real data from a file instead of generating synthetic data.

Usage
-----
    # MATLAB .mat file (FlyAnalysis trial struct)
    python examples/gui_workflow_from_file.py path/to/trial.mat

    # Axon Binary Format .abf file
    python examples/gui_workflow_from_file.py path/to/recording.abf

Requirements
------------
    pip install -e ".[io]"    # from the spikedetect/ directory
"""

import sys

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python gui_workflow_from_file.py <recording_file>")
        print("  Supported formats: .mat, .abf")
        sys.exit(1)

    filepath = sys.argv[1]

    # ---- Load recording -------------------------------------------------
    print(f"Loading {filepath}...")
    if filepath.endswith(".abf"):
        rec = sd.load_abf(filepath)
    else:
        rec = sd.load_recording(filepath)

    fs = rec.sample_rate
    print(f"  {rec.name}: {rec.n_samples} samples, {fs} Hz, {rec.duration:.2f} s")

    params = sd.SpikeDetectionParams.default(fs=fs)

    # If the file already had detection params, offer to reuse them
    if rec.result is not None and rec.result.params.spike_template is not None:
        print("  Found existing detection parameters in file.")
        print("  Using existing template as starting point.")
        params = rec.result.params
    print()

    # ---- Step 1: FilterGUI ----------------------------------------------
    print("Step 1: FilterGUI -- Adjust sliders, then press Enter.")
    filter_gui = FilterGUI(rec.voltage, params)
    params = filter_gui.run()
    print(f"  -> hp={params.hp_cutoff} Hz, lp={params.lp_cutoff} Hz")
    print()

    # ---- Step 2: Filter data for template selection ---------------------
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
    print("Step 2: TemplateSelectionGUI -- Click 3-5 peaks, then Enter.")
    template_gui = TemplateSelectionGUI(filtered_data, params)
    template = template_gui.run()

    if template is not None:
        params.spike_template = template
        print(f"  -> Template length: {len(template)} samples")
    elif params.spike_template is not None:
        print("  -> No new selection, keeping existing template.")
    else:
        print("  ERROR: No template selected and none available. Exiting.")
        sys.exit(1)
    print()

    # ---- Step 4: Initial detection --------------------------------------
    print("Step 3: Running initial detection...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 5: ThresholdGUI -------------------------------------------
    print("Step 4: ThresholdGUI -- Click to adjust thresholds, Enter to accept.")
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
    print(f"  -> distance={params.distance_threshold:.2f}, "
          f"amplitude={params.amplitude_threshold:.3f}")
    print()

    # ---- Step 6: Re-detect with updated thresholds ----------------------
    print("Step 5: Re-running detection...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 7: SpotCheckGUI -------------------------------------------
    print("Step 6: SpotCheckGUI -- y/n/arrows/Enter")
    spotcheck = SpotCheckGUI(rec, result)
    result = spotcheck.run()
    print()

    # ---- Summary --------------------------------------------------------
    print("=" * 50)
    print(result.summary())
    print("=" * 50)


if __name__ == "__main__":
    main()
