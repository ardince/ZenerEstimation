"""
Model optimization infrastructure.
"""

from .candidate import CandidateResult
from .evaluator import OptimizationEvaluator
from .fold import TemporalFold
from .result import OptimizationResult
from .search import GenericOptimizer
from .validation import ExpandingWindowSplitter
from .spaces import (
    ARIMAParameterSpace,
    KalmanParameterSpace,
    LSTMParameterSpace,
    GRUParameterSpace,
    HybridNeuralParameterSpace,
)
from .boundary import BenchmarkBoundary
from .artifact import OptimizationArtifact
from .stability import SeedResult, StabilityResult, FailedSeedResult
from .stability_evaluator import StabilityEvaluator
from .stability_artifact import StabilityArtifact
from .benchmark_artifact import BenchmarkArtifact
from .benchmark_report import BenchmarkReport
from .provenance import BenchmarkProvenance

from .evidence import copy_evidence_files,validate_provenance_references


__all__ = [
    "CandidateResult",
    "OptimizationEvaluator",
    "TemporalFold",
    "OptimizationResult",
    "GenericOptimizer",
    "ExpandingWindowSplitter",
    "ARIMAParameterSpace",
    "BenchmarkBoundary",
    "OptimizationArtifact",
    "KalmanParameterSpace",
    "LSTMParameterSpace",
    "GRUParameterSpace",
    "HybridNeuralParameterSpace",
    "SeedResult",
    "StabilityResult",
    "StabilityEvaluator",
    "FailedSeedResult",
    "StabilityArtifact",
    "BenchmarkArtifact",
    "BenchmarkReport",
    "BenchmarkProvenance",
    "copy_evidence_files",
    "validate_provenance_references"
]