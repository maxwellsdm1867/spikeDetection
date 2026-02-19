"""JSON-based parameter persistence.

Replaces MATLAB setacqpref/getacqpref. Parameters are
stored as JSON files under ``~/.spikedetect/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from spikedetect.models import SpikeDetectionParams

_CONFIG_DIR = Path.home() / ".spikedetect"


def _config_dir() -> Path:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _CONFIG_DIR


def _param_path(input_field: str, fs: float) -> Path:
    """Build the config file path for MATLAB convention.

    MATLAB key: ``Spike_params_{inputToAnalyze}_fs{fs}``
    """
    tag = f"Spike_params_{input_field}_fs{int(fs)}"
    return _config_dir() / f"{tag}.json"


def save_params(
    params: SpikeDetectionParams,
    input_field: str = "voltage_1",
) -> Path:
    """Save detection parameters to JSON.

    Args:
        params: Parameters to persist.
        input_field: Name of the input channel
            (e.g., ``'voltage_1'``).

    Returns:
        Path to the saved JSON file.
    """
    path = _param_path(input_field, params.fs)
    path.write_text(
        json.dumps(params.to_dict(), indent=2)
    )
    return path


def load_params(
    input_field: str = "voltage_1",
    fs: float = 10000,
) -> SpikeDetectionParams | None:
    """Load detection parameters from JSON.

    Args:
        input_field: Name of the input channel.
        fs: Sample rate (used to locate the correct
            config file).

    Returns:
        Loaded parameters, or ``None`` if no config file
        exists.
    """
    path = _param_path(input_field, fs)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return SpikeDetectionParams.from_dict(d)


def list_saved_params() -> list[Path]:
    """List all saved parameter files."""
    config_dir = _config_dir()
    return sorted(
        config_dir.glob("Spike_params_*.json")
    )
