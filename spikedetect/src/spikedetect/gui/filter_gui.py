"""Interactive filter parameter tuning GUI.

Ports the MATLAB ``filter_sliderGUI.m`` function. Provides sliders for
high-pass cutoff, low-pass cutoff, peak threshold, derivative order, and
a polarity toggle button. Filtered data and detected peaks update live.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from spikedetect.gui._widgets import raster_ticks, blocking_wait
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.filtering import filter_data
from spikedetect.pipeline.peaks import find_spike_locations


class FilterGUI:
    """Interactive GUI for tuning filter and peak-detection parameters.

    Args:
        unfiltered_data: Raw 1-D voltage trace.
        params: Initial detection parameters.

    Attributes:
        params: Current parameters (updated by sliders).
        fig: The GUI figure.
    """

    def __init__(
        self,
        unfiltered_data: np.ndarray,
        params: SpikeDetectionParams,
    ) -> None:
        self._unfiltered = np.asarray(unfiltered_data, dtype=np.float64).ravel()
        self.params = deepcopy(params)
        self.fig = None
        self._filtered = None
        self._locs = None

    def run(self) -> SpikeDetectionParams:
        """Display the GUI and block until the user presses Enter.

        Returns:
            Updated parameters reflecting the user's slider
            choices.
        """
        self._apply_filter()
        self._build_figure()
        self._update_plots()

        # Block until keypress
        while True:
            key = blocking_wait(self.fig)
            if key is None or key in ("enter", "return"):
                break

        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

        return self.params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        """Re-filter the data with current params."""
        self._filtered = filter_data(
            self._unfiltered,
            fs=self.params.fs,
            hp_cutoff=self.params.hp_cutoff,
            lp_cutoff=self.params.lp_cutoff,
            diff_order=self.params.diff_order,
            polarity=self.params.polarity,
        )
        self._locs = find_spike_locations(
            self._filtered,
            peak_threshold=self.params.peak_threshold,
            fs=self.params.fs,
            spike_template_width=self.params.spike_template_width,
        )

    def _build_figure(self) -> None:
        """Create the Matplotlib figure with axes, sliders, and buttons."""
        self.fig, (self._ax_unfilt, self._ax_filt) = plt.subplots(
            2, 1, figsize=(12, 9),
            gridspec_kw={"height_ratios": [1, 6], "hspace": 0.15},
        )
        self.fig.subplots_adjust(bottom=0.28, left=0.08, right=0.95)
        self.fig.set_facecolor("white")

        # Unfiltered trace (top)
        self._ax_unfilt.set_title(
            "Adjust filter parameters, then press Enter to accept"
        )

        # Slider axes
        ax_hp = self.fig.add_axes([0.15, 0.02, 0.25, 0.03])
        ax_lp = self.fig.add_axes([0.15, 0.07, 0.25, 0.03])
        ax_thresh = self.fig.add_axes([0.55, 0.02, 0.30, 0.03])

        # HP slider (0.5 -- 1000 Hz)
        self._sl_hp = Slider(
            ax_hp, "HP (Hz)", 0.5, 1000.0,
            valinit=self.params.hp_cutoff, valstep=0.5,
        )
        # LP slider (0.11 -- 1000 Hz)
        self._sl_lp = Slider(
            ax_lp, "LP (Hz)", 0.11, 1000.0,
            valinit=self.params.lp_cutoff, valstep=0.5,
        )
        # Peak threshold on log10 scale
        log_thresh = np.log10(max(self.params.peak_threshold, 1e-10))
        self._sl_thresh = Slider(
            ax_thresh, "Peak thresh (log10)", -10, -1,
            valinit=log_thresh,
        )

        # Diff radio buttons
        ax_diff = self.fig.add_axes([0.02, 0.12, 0.06, 0.10])
        self._radio_diff = RadioButtons(
            ax_diff, ("0", "1", "2"),
            active=self.params.diff_order,
        )
        ax_diff.set_title("Diff", fontsize=9)

        # Polarity toggle button
        ax_pol = self.fig.add_axes([0.02, 0.02, 0.06, 0.04])
        pol_label = f"Pol: {self.params.polarity:+d}"
        self._btn_pol = Button(ax_pol, pol_label)

        # Connect callbacks
        self._sl_hp.on_changed(self._on_slider_change)
        self._sl_lp.on_changed(self._on_slider_change)
        self._sl_thresh.on_changed(self._on_slider_change)
        self._radio_diff.on_clicked(self._on_diff_change)
        self._btn_pol.on_clicked(self._on_polarity_toggle)

    def _on_slider_change(self, _val) -> None:
        """Callback for any slider change."""
        self.params.hp_cutoff = self._sl_hp.val
        self.params.lp_cutoff = self._sl_lp.val
        self.params.peak_threshold = 10 ** self._sl_thresh.val
        self._apply_filter()
        self._update_plots()

    def _on_diff_change(self, label: str) -> None:
        """Callback for diff order radio button."""
        self.params.diff_order = int(label)
        self._apply_filter()
        self._update_plots()

    def _on_polarity_toggle(self, _event) -> None:
        """Callback for polarity toggle button."""
        self.params.polarity *= -1
        self._btn_pol.label.set_text(f"Pol: {self.params.polarity:+d}")
        self._apply_filter()
        self._update_plots()

    def _update_plots(self) -> None:
        """Redraw the filtered data and peak markers."""
        n = len(self._unfiltered)

        # Top axis: unfiltered with raster ticks
        self._ax_unfilt.cla()
        self._ax_unfilt.plot(
            np.arange(n), self._unfiltered,
            color=(0.85, 0.325, 0.098),
        )
        if len(self._locs) > 0:
            y_top = np.max(self._unfiltered) + 0.02 * np.ptp(self._unfiltered)
            raster_ticks(self._ax_unfilt, self._locs, y_top)
        self._ax_unfilt.set_xlim(0, n)
        self._ax_unfilt.set_title(
            "Adjust filter parameters, then press Enter to accept"
        )

        # Bottom axis: filtered with peaks
        self._ax_filt.cla()
        filt = self._filtered
        self._ax_filt.plot(np.arange(n), filt, "k", linewidth=0.5)
        if len(self._locs) > 0:
            self._ax_filt.plot(self._locs, filt[self._locs], "ro", markersize=4)
        # Threshold line
        self._ax_filt.axhline(
            self.params.peak_threshold, color=(0.8, 0.8, 0.8),
            linestyle="--", linewidth=1,
        )
        self._ax_filt.set_xlim(0, n)
        if len(filt) > 0:
            self._ax_filt.set_ylim(np.min(filt), np.max(filt))

        self.fig.canvas.draw_idle()
