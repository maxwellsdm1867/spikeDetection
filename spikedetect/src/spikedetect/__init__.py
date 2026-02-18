"""Spike detection and sorting for electrophysiology recordings.

Converted from MATLAB codebase using DTW template matching for spike
classification with interactive GUI tools for parameter tuning.
"""

from spikedetect.models import Recording, SpikeDetectionParams, SpikeDetectionResult
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
    "DTW",
    "InflectionPointDetector",
    "PeakFinder",
    "SignalFilter",
    "SpikeClassifier",
    "TemplateMatcher",
    "WaveformProcessor",
]

__version__ = "0.1.0"
