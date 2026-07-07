"""
ZenerEstimation

Scientific framework for battery voltage forecasting
and Remaining Useful Life estimation.
"""

from .version import VERSION
from .pipeline import ForecastPipeline

__version__ = VERSION

__all__ = [
    "ForecastPipeline",
]