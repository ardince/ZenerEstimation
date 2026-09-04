"""
Canonical dataset processing framework.
"""

from .processor import DatasetProcessor
from .result import DatasetProcessingResult
from .writer import (
    ProcessedDatasetPaths,
    ProcessedDatasetWriter,
)

__all__ = [
    "DatasetProcessor",
    "DatasetProcessingResult",
    "ProcessedDatasetPaths",
    "ProcessedDatasetWriter",
]