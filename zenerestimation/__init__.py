"""
ZenerEstimation

Scientific framework for degradation modelling and forecasting.
"""

from .pipeline import ForecastPipeline
from .version import __version__

from .data import BatteryDataset

__all__ = [
    "ForecastPipeline",
    "BatteryDataset",
    "__version__",
]