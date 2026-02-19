"""Load ABF (Axon Binary Format) electrophysiology files.

Uses the ``pyabf`` library to read ABF files into
:class:`~spikedetect.models.Recording` objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spikedetect.models import Recording


def load_abf(path: str | Path) -> Recording:
    """Load an ABF file into a Recording object.

    Args:
        path: Path to the .abf file.

    Returns:
        The loaded recording with voltage data from the
        first channel and current from the second channel
        (if available).

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If ``pyabf`` is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ABF file not found: {path}"
        )

    try:
        import pyabf
    except ImportError:
        raise ImportError(
            "pyabf is required to load ABF files. "
            "Install it with: pip install pyabf"
        )

    abf = pyabf.ABF(str(path))

    # Read first channel (voltage)
    abf.setSweep(0, channel=0)
    voltage = np.array(abf.sweepY, dtype=np.float64)

    # Read second channel (current) if available
    current = None
    if abf.channelCount > 1:
        abf.setSweep(0, channel=1)
        current = np.array(
            abf.sweepY, dtype=np.float64
        )

    sample_rate = float(abf.dataRate)

    return Recording(
        name=str(path),
        voltage=voltage,
        sample_rate=sample_rate,
        current=current,
        metadata={
            "abf_id": abf.abfID,
            "protocol": abf.protocol,
            "channel_count": abf.channelCount,
            "sweep_count": abf.sweepCount,
        },
    )
