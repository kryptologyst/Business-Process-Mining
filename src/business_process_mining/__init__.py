"""Business Process Mining and Optimization Package.

This package provides tools for analyzing event logs to discover, monitor, 
and improve real business processes. It includes process discovery algorithms,
conformance checking, and process optimization capabilities.

DISCLAIMER: This software is for research and educational purposes only.
It should not be used for automated decision-making without human review.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .data.event_log import EventLog, EventLogGenerator
from .mining.discovery import ProcessDiscovery
from .mining.conformance import ConformanceChecker
from .eval.metrics import ProcessMetrics
from .viz.visualization import ProcessVisualizer

__all__ = [
    "EventLog",
    "EventLogGenerator", 
    "ProcessDiscovery",
    "ConformanceChecker",
    "ProcessMetrics",
    "ProcessVisualizer",
]
