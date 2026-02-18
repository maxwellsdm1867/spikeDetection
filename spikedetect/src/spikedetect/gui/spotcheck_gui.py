"""Interactive spike-by-spike review GUI.

Ports the MATLAB ``spikeSpotCheck.m`` function. Allows the user to step
through each detected spike, accept or reject it, adjust its position,
and add or remove spikes from the detection result.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spikedetect.gui._widgets import raster_ticks, blocking_wait
from spikedetect.models import Recording, SpikeDetectionParams, SpikeDetectionResult
from spikedetect.pipeline.filtering import filter_data
from spikedetect.pipeline.peaks import find_spike_locations
from spikedetect.pipeline.template import match_template
from spikedetect.utils import smooth, smooth_and_differentiate


class SpotCheckGUI:
    """Interactive GUI for reviewing spikes one by one.

    Parameters
    ----------
    recording : Recording
        The electrophysiology recording.
    result : SpikeDetectionResult
        Detection result to review and potentially modify.

    Attributes
    ----------
    result : SpikeDetectionResult
        The (possibly modified) detection result.
    fig : matplotlib.figure.Figure
        The GUI figure.
    """

    def __init__(self, recording: Recording, result: SpikeDetectionResult) -> None:
        self._recording = recording
        self.result = deepcopy(result)
        self.fig = None
        self._spike_idx = 0
        self._accepted: np.ndarray | None = None
        self._direction = 1

    def run(self) -> SpikeDetectionResult:
        """Display the GUI and block until the user finishes reviewing.

        Keyboard controls:
            y       : Accept spike and move to next
            n       : Reject spike (remove) and move to next
            right   : Shift spike position right (+10 samples, +1 with shift)
            left    : Shift spike position left (-10 samples, -1 with shift)
            tab     : Skip to next spike without decision
            enter   : Finish review

        Returns
        -------
        SpikeDetectionResult
            Updated result with spot_checked set to True.
        """
        self._setup()
        self._build_figure()

        if self.result.n_spikes == 0:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
            self.result.spot_checked = True
            return self.result

        self._show_current_spike()

        while True:
            key = blocking_wait(self.fig)
            if key is None or key in ("enter", "return"):
                break

            action = self._handle_key(key)
            if action == "done":
                break

        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

        # Build final result from accepted spikes
        self.result.spike_times = np.sort(
            self._spikes[self._accepted]
        ).astype(np.int64)
        self.result.spot_checked = True
        return self.result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Prepare internal data structures."""
        params = self.result.params
        voltage = self._recording.voltage
        fs = params.fs

        # Filter the data
        self._filtered = filter_data(
            voltage, fs=fs,
            hp_cutoff=params.hp_cutoff,
            lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order,
            polarity=params.polarity,
        )

        # Working copy of spike positions
        self._spikes = self.result.spike_times.copy()
        self._accepted = np.ones(len(self._spikes), dtype=bool)
        self._spike_idx = 0

        # Compute context window size from template width
        stw = params.spike_template_width
        half = stw // 2
        self._window = np.arange(-half, half + 1)
        self._context_window_half = half * 4  # 4x the spike width

        # Compute mean spike waveform and 2nd derivative
        if params.spike_template is not None and len(self._spikes) > 0:
            self._compute_mean_waveform()

    def _compute_mean_waveform(self) -> None:
        """Compute the mean spike waveform and its 2nd derivative."""
        params = self.result.params
        stw = params.spike_template_width
        half = stw // 2
        window = self._window
        smth_w = max(round(params.fs / 2000), 1)

        waveforms = []
        for s in self._spikes:
            if s + window[0] >= 0 and s + window[-1] < len(self._recording.voltage):
                waveforms.append(self._recording.voltage[s + window])

        if len(waveforms) > 0:
            mean_wf = np.mean(waveforms, axis=0)
            mean_wf = mean_wf - np.min(mean_wf)
            mx = np.max(mean_wf)
            if mx > 0:
                mean_wf = mean_wf / mx
            self._mean_waveform = smooth(mean_wf - mean_wf[0], smth_w)
            self._mean_2d = smooth_and_differentiate(self._mean_waveform, smth_w)
        else:
            self._mean_waveform = np.zeros(len(window))
            self._mean_2d = np.zeros(len(window))

    def _build_figure(self) -> None:
        """Create the figure with subplots for review."""
        self.fig = plt.figure(figsize=(16, 7))
        self.fig.set_facecolor("white")
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.4, wspace=0.3,
                      height_ratios=[2, 1, 1, 4])

        # Top: unfiltered trace with raster ticks
        self._ax_trace = self.fig.add_subplot(gs[0, :])
        self._ax_trace.set_title("Unfiltered trace with spike ticks")

        # Second row: filtered trace
        self._ax_filt = self.fig.add_subplot(gs[1, :])

        # Third row: current channel (optional)
        self._ax_current = self.fig.add_subplot(gs[2, :])

        # Bottom left: DTW scatter
        self._ax_hist = self.fig.add_subplot(gs[3, 0])
        self._ax_hist.set_xlabel("DTW Distance")
        self._ax_hist.set_ylabel("Amplitude")

        # Bottom center: spike context (unfiltered)
        self._ax_spike = self.fig.add_subplot(gs[3, 1])
        self._ax_spike.set_title("Is this a spike? (y/n) Arrows to adjust")

        # Bottom right: filtered context
        self._ax_squig = self.fig.add_subplot(gs[3, 2])
        self._ax_squig.set_title("Filtered context")

        # Draw the full traces
        voltage = self._recording.voltage
        n = len(voltage)
        t = np.arange(n) / self.result.params.fs

        self._ax_trace.plot(t, voltage, color=(0.85, 0.325, 0.098), linewidth=0.5)
        if len(self._spikes) > 0:
            y_top = np.max(voltage) + 0.02 * np.ptp(voltage)
            raster_ticks(self._ax_trace, self._spikes / self.result.params.fs, y_top)
        self._ax_trace.set_xlim(t[0], t[-1])

        filt_mean = self._filtered - np.mean(self._filtered)
        self._ax_filt.plot(t, filt_mean, color=(0.0, 0.45, 0.74), linewidth=0.5)
        self._ax_filt.set_xlim(t[0], t[-1])

        if self._recording.current is not None:
            self._ax_current.plot(
                t, self._recording.current, color=(0.74, 0, 0), linewidth=0.5,
            )
            self._ax_current.set_xlim(t[0], t[-1])
        else:
            self._ax_current.set_visible(False)

    def _show_current_spike(self) -> None:
        """Update the bottom panels to show the current spike."""
        if self._spike_idx < 0 or self._spike_idx >= len(self._spikes):
            return

        params = self.result.params
        fs = params.fs
        spike = self._spikes[self._spike_idx]
        voltage = self._recording.voltage
        n = len(voltage)
        ctx_half = self._context_window_half

        # Context boundaries (clamp to valid range)
        ctx_start = max(0, spike - ctx_half)
        ctx_end = min(n, spike + ctx_half)

        t_ctx = np.arange(ctx_start, ctx_end) / fs

        # Spike context view
        self._ax_spike.cla()
        self._ax_spike.plot(t_ctx, voltage[ctx_start:ctx_end],
                           color=(0.49, 0.18, 0.56), linewidth=0.8)

        # Highlight the spike window
        win_start = max(0, spike + self._window[0])
        win_end = min(n, spike + self._window[-1] + 1)
        t_win = np.arange(win_start, win_end) / fs
        self._ax_spike.plot(t_win, voltage[win_start:win_end],
                           color=(0, 0, 0), linewidth=1.5)

        # Mean waveform overlay
        if hasattr(self, "_mean_waveform"):
            amp = np.mean(np.abs(voltage[win_start:win_end] - voltage[spike]))
            scale = max(amp, 0.01)
            mean_scaled = self._mean_waveform * scale + voltage[spike]
            if len(mean_scaled) == len(t_win):
                self._ax_spike.plot(
                    t_win, mean_scaled,
                    color=(0.4, 0.3, 1.0), linewidth=2, alpha=0.7,
                )
            # 2nd derivative overlay
            smth_start = round(fs / 2000)
            smth_end = len(self._mean_2d) - smth_start
            if smth_end > smth_start + 2:
                region_2d = self._mean_2d[smth_start + 1 : smth_end - 1]
                if np.max(np.abs(region_2d)) > 0:
                    region_scaled = (
                        region_2d / np.max(np.abs(region_2d)) * scale + voltage[spike]
                    )
                    t_2d = np.arange(
                        win_start + smth_start + 1, win_start + smth_end - 1
                    ) / fs
                    if len(t_2d) == len(region_scaled):
                        self._ax_spike.plot(
                            t_2d, region_scaled,
                            color=(0, 0.8, 0.4), linewidth=2, alpha=0.7,
                        )

        # Vertical spike marker
        self._ax_spike.axvline(
            spike / fs, color=(1, 0, 0), linewidth=1, linestyle="--",
        )

        is_accepted = self._accepted[self._spike_idx]
        status = "accepted" if is_accepted else "REJECTED"
        self._ax_spike.set_title(
            f"Spike {self._spike_idx + 1}/{len(self._spikes)} [{status}] "
            f"(y=accept, n=reject, arrows=adjust)"
        )

        self._ax_spike.set_xlim(t_ctx[0], t_ctx[-1])

        # Filtered context view
        self._ax_squig.cla()
        self._ax_squig.plot(
            t_ctx, self._filtered[ctx_start:ctx_end],
            color=(0.49, 0.18, 0.56), linewidth=0.8,
        )
        if params.spike_template is not None:
            tmpl = params.spike_template
            tmpl_scaled = tmpl / np.max(np.abs(tmpl)) * np.max(
                np.abs(self._filtered[ctx_start:ctx_end])
            )
            half = len(tmpl) // 2
            # Use uncorrected spike location for template overlay
            uc_spike = spike
            if self._spike_idx < len(self.result.spike_times_uncorrected):
                uc_spike = self.result.spike_times_uncorrected[self._spike_idx]
            tmpl_start = max(0, uc_spike - half)
            tmpl_end = min(n, uc_spike + half + 1)
            t_tmpl = np.arange(tmpl_start, tmpl_end) / fs
            tmpl_len = tmpl_end - tmpl_start
            if tmpl_len == len(tmpl):
                self._ax_squig.plot(
                    t_tmpl, tmpl_scaled,
                    color=(0.85, 0.325, 0.098), linewidth=1,
                )
        self._ax_squig.set_xlim(t_ctx[0], t_ctx[-1])
        self._ax_squig.set_title("Filtered context with template")

        # Update DTW scatter -- highlight current spike
        self._ax_hist.cla()
        self._ax_hist.set_xlabel("DTW Distance")
        self._ax_hist.set_title("Click spike on scatter (not implemented)")

        self.fig.canvas.draw_idle()

    def _handle_key(self, key: str) -> str | None:
        """Process a keypress and return 'done' to finish, or None."""
        if key == "y":
            self._accepted[self._spike_idx] = True
            self._advance()
        elif key == "n":
            self._accepted[self._spike_idx] = False
            self._advance()
        elif key == "right":
            self._spikes[self._spike_idx] += 10
            self._show_current_spike()
        elif key == "shift+right":
            self._spikes[self._spike_idx] += 1
            self._show_current_spike()
        elif key == "left":
            self._spikes[self._spike_idx] -= 10
            self._show_current_spike()
        elif key == "shift+left":
            self._spikes[self._spike_idx] -= 1
            self._show_current_spike()
        elif key == "tab":
            self._advance()
        elif key == "shift+tab":
            self._direction = -1
            self._advance()
            self._direction = 1
        elif key in ("enter", "return"):
            return "done"
        return None

    def _advance(self) -> None:
        """Move to the next (or previous) spike."""
        self._spike_idx += self._direction
        if self._spike_idx < 0:
            self._spike_idx = 0
        elif self._spike_idx >= len(self._spikes):
            self._spike_idx = len(self._spikes) - 1
        self._show_current_spike()
