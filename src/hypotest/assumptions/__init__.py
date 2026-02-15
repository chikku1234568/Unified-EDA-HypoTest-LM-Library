"""
Assumption checking and validation.
"""

from .base import AssumptionResult, AssumptionChecker
from .normality import NormalityChecker
from .variance import VarianceChecker

__all__ = [
    "AssumptionResult",
    "AssumptionChecker",
    "NormalityChecker",
    "VarianceChecker",
]