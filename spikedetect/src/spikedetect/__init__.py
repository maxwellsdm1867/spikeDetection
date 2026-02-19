"""Spike detection and sorting for electrophysiology recordings.

Converted from MATLAB codebase using DTW template matching for spike
classification with interactive GUI tools for parameter tuning.

Quick start
-----------
>>> import spikedetect as sd
>>> rec = sd.load_recording("trial.mat")
>>> params = sd.SpikeDetectionParams.default(fs=rec.sample_rate)
>>> # Set a spike template (from GUI or a previous run):
>>> # params.spike_template = my_template
>>> result = sd.detect_spikes(rec, params)
>>> print(result.summary())

To see pipeline progress, enable logging::

    import logging
    logging.basicConfig(level=logging.INFO)
"""

from spikedetect.io import load_recording, load_abf, save_result
from spikedetect.models import (
    Recording,
    SpikeDetectionParams,
    SpikeDetectionResult,
)
from spikedetect.pipeline.classify import SpikeClassifier
from spikedetect.pipeline.detect import SpikeDetector, detect_spikes
from spikedetect.pipeline.dtw import DTW
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.inflection import InflectionPointDetector
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher
from spikedetect.utils import WaveformProcessor

__all__ = [
    "Recording",
    "SpikeDetectionParams",
    "SpikeDetectionResult",
    "SpikeDetector",
    "detect_spikes",
    "load_recording",
    "load_abf",
    "save_result",
    "DTW",
    "InflectionPointDetector",
    "PeakFinder",
    "SignalFilter",
    "SpikeClassifier",
    "TemplateMatcher",
    "WaveformProcessor",
]

__version__ = "0.1.0"
