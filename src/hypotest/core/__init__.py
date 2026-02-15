"""
Core data structures for hypothesis testing.
"""

from .dataset import Dataset
from .result import TestResult, AnalysisResult

__all__ = ["Dataset", "TestResult", "AnalysisResult"]