"""Interactive Matplotlib GUI components for spike detection parameter tuning."""

from spikedetect.gui.filter_gui import FilterGUI
from spikedetect.gui.template_gui import TemplateSelectionGUI
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.gui.spotcheck_gui import SpotCheckGUI
from spikedetect.gui.workflow import InteractiveWorkflow

__all__ = [
    "FilterGUI",
    "InteractiveWorkflow",
    "TemplateSelectionGUI",
    "ThresholdGUI",
    "SpotCheckGUI",
]
