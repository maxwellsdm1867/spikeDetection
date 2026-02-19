#!/usr/bin/env python
"""Complete interactive GUI workflow using synthetic data.

This script walks through all 4 spikedetect GUIs in order:
  1. FilterGUI      -- tune bandpass filter parameters
  2. TemplateSelectionGUI -- click peaks to build a spike template
  3. ThresholdGUI    -- adjust DTW distance and amplitude thresholds
  4. SpotCheckGUI    -- review individual spikes

No .mat or .abf file is needed -- the script generates a synthetic
recording with embedded spikes so you can try the full workflow
immediately after installing spikedetect.

Usage
-----
    python examples/gui_workflow.py

Requirements
------------
    pip install -e .          # from the spikedetect/ directory
"""

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
    """Generate a synthetic recording with embedded Gaussian spikes.

    Returns
    -------
    recording : sd.Recording
        A Recording object with embedded spikes in low noise.
    true_positions : np.ndarray
        Sample indices where spikes were inserted.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration)
    voltage = rng.normal(0, 0.001, n_samples)

    # Space spikes evenly, avoiding the edges
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
    # ---- Generate synthetic data ----------------------------------------
    print("Generating synthetic recording with embedded spikes...")
    rec, true_positions = make_synthetic_recording()
    fs = rec.sample_rate
    params = sd.SpikeDetectionParams.default(fs=fs)
    print(f"  {rec.n_samples} samples, {fs} Hz, {rec.duration:.1f} s")
    print(f"  {len(true_positions)} embedded spikes")
    print()

    # ---- Step 1: FilterGUI ----------------------------------------------
    # The FilterGUI lets you adjust high-pass cutoff, low-pass cutoff,
    # peak threshold, differentiation order, and polarity with live
    # preview. Adjust sliders until peaks align with spikes, then press
    # Enter to accept.
    print("Step 1: FilterGUI -- Tune filter parameters")
    print("  Adjust sliders, then press Enter to accept.")
    filter_gui = FilterGUI(rec.voltage, params)
    params = filter_gui.run()
    print(f"  -> hp={params.hp_cutoff} Hz, lp={params.lp_cutoff} Hz, "
          f"diff={params.diff_order}, pol={params.polarity:+d}")
    print()

    # ---- Step 2: Filter the data ----------------------------------------
    # Apply the chosen filter parameters to get the filtered signal.
    # This is needed for TemplateSelectionGUI.
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
    # Click on 3-5 clear spike peaks (they turn green when selected).
    # The GUI averages your selections to build the spike template.
    # Press Enter when done.
    print("Step 2: TemplateSelectionGUI -- Click peaks to build a template")
    print("  Click 3-5 clear peaks, then press Enter.")
    template_gui = TemplateSelectionGUI(filtered_data, params)
    template = template_gui.run()

    if template is None:
        print("  No spikes selected -- using auto-generated template.")
        # Fallback: extract template from the first known spike
        stw = params.spike_template_width
        half = stw // 2
        spike_in_trimmed = true_positions[0] - start_point
        template = filtered_data[spike_in_trimmed - half:spike_in_trimmed + half + 1]

    params.spike_template = template
    print(f"  -> Template length: {len(template)} samples")
    print()

    # ---- Step 4: Run initial detection ----------------------------------
    print("Step 3: Running initial spike detection...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 5: ThresholdGUI -------------------------------------------
    # To use the ThresholdGUI, we need the TemplateMatcher result (which
    # contains DTW distances and amplitudes for the scatter plot).
    # We re-run the pipeline stages manually to get match_result.
    print("Step 4: ThresholdGUI -- Adjust distance/amplitude thresholds")
    print("  Click on the scatter plot to move the threshold line.")
    print("  Press 'b' to toggle between distance and amplitude.")
    print("  Press Enter to accept.")

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

    # ---- Step 6: Re-run detection with updated thresholds ---------------
    print("Step 5: Re-running detection with updated thresholds...")
    result = sd.detect_spikes(rec, params)
    print(f"  -> Found {result.n_spikes} spikes")
    print()

    # ---- Step 7: SpotCheckGUI -------------------------------------------
    # Step through each spike one by one:
    #   y / Enter  -- accept
    #   n          -- reject
    #   left/right -- shift spike position
    #   tab        -- skip to next
    #   Enter      -- finish review
    print("Step 6: SpotCheckGUI -- Review spikes one by one")
    print("  y=accept, n=reject, arrows=adjust, Enter=done")
    spotcheck = SpotCheckGUI(rec, result)
    result = spotcheck.run()
    print()

    # ---- Summary --------------------------------------------------------
    print("=" * 50)
    print(result.summary())
    print("=" * 50)


if __name__ == "__main__":
    main()
