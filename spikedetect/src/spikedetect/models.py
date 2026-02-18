"""Data models for spike detection.

Converted from MATLAB structs: ``vars`` -> SpikeDetectionParams,
``trial`` -> Recording, detection outputs -> SpikeDetectionResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpikeDetectionParams:
    """Persistent detection parameters (replaces MATLAB ``vars`` struct).

    Only persistent parameters live here. Transient data (filtered_data,
    locs, etc.) exist as local variables in pipeline functions.

    Original MATLAB fields -> Python attributes:
        fs                  -> fs
        spikeTemplateWidth  -> spike_template_width
        hp_cutoff           -> hp_cutoff
        lp_cutoff           -> lp_cutoff
        diff                -> diff_order
        peak_threshold      -> peak_threshold
        Distance_threshold  -> distance_threshold
        Amplitude_threshold -> amplitude_threshold
        spikeTemplate       -> spike_template
        polarity            -> polarity
        likelyiflpntpeak    -> likely_inflection_point_peak
        lastfilename        -> last_filename
    """

    fs: float
    spike_template_width: int = 0
    hp_cutoff: float = 200.0
    lp_cutoff: float = 800.0
    diff_order: int = 1
    peak_threshold: float = 5.0
    distance_threshold: float = 15.0
    amplitude_threshold: float = 0.2
    spike_template: np.ndarray | None = None
    polarity: int = 1
    likely_inflection_point_peak: int | None = None
    last_filename: str = ""

    def __post_init__(self) -> None:
        if self.spike_template_width == 0:
            self.spike_template_width = round(0.005 * self.fs) + 1
        if self.spike_template is not None:
            self.spike_template = np.asarray(self.spike_template, dtype=np.float64)

    @classmethod
    def default(cls, fs: float = 10000) -> SpikeDetectionParams:
        """Create params with sensible defaults for a given sample rate."""
        return cls(fs=fs)

    def validate(self) -> SpikeDetectionParams:
        """Validate and clean parameters (replaces cleanUpSpikeVarsStruct).

        Returns self for chaining.
        """
        if self.fs <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.fs}")
        if self.hp_cutoff <= 0:
            raise ValueError(f"HP cutoff must be positive, got {self.hp_cutoff}")
        if self.lp_cutoff <= 0:
            raise ValueError(f"LP cutoff must be positive, got {self.lp_cutoff}")
        if self.hp_cutoff >= self.fs / 2:
            raise ValueError(
                f"HP cutoff ({self.hp_cutoff}) must be below Nyquist ({self.fs / 2})"
            )
        if self.lp_cutoff >= self.fs / 2:
            raise ValueError(
                f"LP cutoff ({self.lp_cutoff}) must be below Nyquist ({self.fs / 2})"
            )
        if self.diff_order not in (0, 1, 2):
            raise ValueError(f"diff_order must be 0, 1, or 2, got {self.diff_order}")
        if self.polarity not in (-1, 1):
            raise ValueError(f"polarity must be -1 or 1, got {self.polarity}")
        if self.spike_template is not None:
            self.spike_template_width = len(self.spike_template)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = {
            "fs": self.fs,
            "spike_template_width": self.spike_template_width,
            "hp_cutoff": self.hp_cutoff,
            "lp_cutoff": self.lp_cutoff,
            "diff_order": self.diff_order,
            "peak_threshold": self.peak_threshold,
            "distance_threshold": self.distance_threshold,
            "amplitude_threshold": self.amplitude_threshold,
            "polarity": self.polarity,
            "last_filename": self.last_filename,
        }
        if self.spike_template is not None:
            d["spike_template"] = self.spike_template.tolist()
        if self.likely_inflection_point_peak is not None:
            d["likely_inflection_point_peak"] = self.likely_inflection_point_peak
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpikeDetectionParams:
        """Deserialize from a dict (e.g., loaded from JSON)."""
        template = d.get("spike_template")
        if template is not None:
            template = np.asarray(template, dtype=np.float64)
        return cls(
            fs=d["fs"],
            spike_template_width=d.get("spike_template_width", 0),
            hp_cutoff=d.get("hp_cutoff", 200.0),
            lp_cutoff=d.get("lp_cutoff", 800.0),
            diff_order=d.get("diff_order", 1),
            peak_threshold=d.get("peak_threshold", 5.0),
            distance_threshold=d.get("distance_threshold", 15.0),
            amplitude_threshold=d.get("amplitude_threshold", 0.2),
            spike_template=template,
            polarity=d.get("polarity", 1),
            likely_inflection_point_peak=d.get("likely_inflection_point_peak"),
            last_filename=d.get("last_filename", ""),
        )


@dataclass
class Recording:
    """A single electrophysiology recording (replaces MATLAB ``trial`` struct).

    Original MATLAB fields -> Python attributes:
        trial.name      -> name
        trial.voltage_1 -> voltage
        trial.params.sampratein -> sample_rate
        trial.current_2 -> current (optional)
    """

    name: str
    voltage: np.ndarray
    sample_rate: float
    current: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: SpikeDetectionResult | None = None

    def __post_init__(self) -> None:
        self.voltage = np.asarray(self.voltage, dtype=np.float64).ravel()
        if self.current is not None:
            self.current = np.asarray(self.current, dtype=np.float64).ravel()


@dataclass
class SpikeDetectionResult:
    """Output of the spike detection pipeline.

    Original MATLAB fields -> Python attributes:
        trial.spikes              -> spike_times (sample indices)
        trial.spikes_uncorrected  -> spike_times_uncorrected
        trial.spikeDetectionParams -> params
        trial.spikeSpotChecked    -> spot_checked
    """

    spike_times: np.ndarray
    spike_times_uncorrected: np.ndarray
    params: SpikeDetectionParams
    spot_checked: bool = False

    def __post_init__(self) -> None:
        self.spike_times = np.asarray(self.spike_times, dtype=np.int64)
        self.spike_times_uncorrected = np.asarray(self.spike_times_uncorrected, dtype=np.int64)

    @property
    def n_spikes(self) -> int:
        return len(self.spike_times)

    @property
    def spike_times_seconds(self) -> np.ndarray:
        """Spike times in seconds."""
        return self.spike_times.astype(np.float64) / self.params.fs
