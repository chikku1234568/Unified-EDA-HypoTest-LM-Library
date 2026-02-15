"""
Configuration management for hypothesis testing.

This module provides a global, thread‑safe configuration system for the hypotest
library. Users can explicitly set a configuration via `initialize()` or rely on
automatic defaults.
"""

from .settings import HypotestConfig
from .manager import initialize, get_config, is_initialized

__all__ = [
    "HypotestConfig",
    "initialize",
    "get_config",
    "is_initialized",
]