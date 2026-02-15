"""
Internal test utilities (not to be confused with the top-level tests).
"""

from .base import StatisticalTest, TestMetadata
from .registry import TestRegistry, register_test

__all__ = [
    "StatisticalTest",
    "TestMetadata",
    "TestRegistry",
    "register_test",
]