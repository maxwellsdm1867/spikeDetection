"""Interactive Matplotlib GUI components for spike detection parameter tuning."""

from spikedetect.gui.filter_gui import FilterGUI
from spikedetect.gui.template_gui import TemplateSelectionGUI
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.gui.spotcheck_gui import SpotCheckGUI

__all__ = [
    "FilterGUI",
    "TemplateSelectionGUI",
    "ThresholdGUI",
    "SpotCheckGUI",
]
