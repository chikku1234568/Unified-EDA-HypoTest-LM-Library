"""
Statistical analysis and type detection.
"""

from .type_detector import TypeDetector, VariableType
from .statistics import (
    compute_statistics,
    correlation,
    Statistics,
    StatisticsCalculator,
)

__all__ = [
    "TypeDetector",
    "VariableType",
    "compute_statistics",
    "correlation",
    "Statistics",
    "StatisticsCalculator",
]