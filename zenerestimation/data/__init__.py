"""
Data package.

Provides the central BatteryDataset object together with
data loading, validation, interpolation and preprocessing
utilities.
"""

from .dataset import BatteryDataset
from .windows import WindowGenerator

__all__ = [
    "BatteryDataset",
    "WindowGenerator",
]