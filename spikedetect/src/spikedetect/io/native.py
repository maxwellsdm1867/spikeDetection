"""HDF5 native format for efficient storage of recordings and results.

Provides a compact, fast-loading format using h5py with voltage/current
stored as HDF5 datasets and parameters serialized as a JSON attribute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spikedetect.models import Recording, SpikeDetectionParams, SpikeDetectionResult


def save_native(path: str | Path, recording: Recording) -> None:
    """Save a recording and its results to HDF5 native format.

    Parameters
    ----------
    path : str or Path
        Output file path (typically with ``.h5`` extension).
    recording : Recording
        The recording to save, optionally with detection results.
    """
    import h5py

    path = Path(path)

    with h5py.File(path, "w") as f:
        # Recording metadata
        f.attrs["name"] = recording.name
        f.attrs["sample_rate"] = recording.sample_rate

        # Voltage data
        f.create_dataset(
            "voltage", data=recording.voltage, compression="gzip", compression_opts=4
        )

        # Current data (optional)
        if recording.current is not None:
            f.create_dataset(
                "current",
                data=recording.current,
                compression="gzip",
                compression_opts=4,
            )

        # Extra metadata
        if recording.metadata:
            f.attrs["metadata"] = json.dumps(recording.metadata)

        # Spike detection results
        if recording.result is not None:
            res = recording.result
            rg = f.create_group("result")
            rg.create_dataset("spike_times", data=res.spike_times.astype(np.int64))
            rg.create_dataset(
                "spike_times_uncorrected",
                data=res.spike_times_uncorrected.astype(np.int64),
            )
            rg.attrs["spot_checked"] = res.spot_checked
            rg.attrs["params"] = json.dumps(res.params.to_dict())


def load_native(path: str | Path) -> Recording:
    """Load a recording from HDF5 native format.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5`` file.

    Returns
    -------
    Recording
        The loaded recording with any stored detection results.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Native file not found: {path}")

    with h5py.File(path, "r") as f:
        name = f.attrs["name"]
        sample_rate = float(f.attrs["sample_rate"])

        voltage = np.asarray(f["voltage"], dtype=np.float64)

        current = None
        if "current" in f:
            current = np.asarray(f["current"], dtype=np.float64)

        metadata = {}
        if "metadata" in f.attrs:
            metadata = json.loads(f.attrs["metadata"])

        result = None
        if "result" in f:
            rg = f["result"]
            params = SpikeDetectionParams.from_dict(json.loads(rg.attrs["params"]))
            result = SpikeDetectionResult(
                spike_times=np.asarray(rg["spike_times"], dtype=np.int64),
                spike_times_uncorrected=np.asarray(
                    rg["spike_times_uncorrected"], dtype=np.int64
                ),
                params=params,
                spot_checked=bool(rg.attrs["spot_checked"]),
            )

    return Recording(
        name=name,
        voltage=voltage,
        sample_rate=sample_rate,
        current=current,
        metadata=metadata,
        result=result,
    )
