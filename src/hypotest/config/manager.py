"""
Singleton configuration manager for hypotest.

This module provides a global, thread‑safe manager that holds the library's runtime
configuration. The configuration can be explicitly set via `initialize()` or will
be lazily initialized with defaults when first accessed.
"""

import threading
from typing import Optional
from .settings import HypotestConfig


class ConfigManager:
    """
    Singleton manager for hypotest configuration.

    This class ensures a single, globally accessible configuration instance.
    It is thread‑safe and supports lazy initialization with defaults.

    Attributes
    ----------
    _instance : Optional[ConfigManager]
        Class‑level singleton instance.
    _lock : threading.Lock
        Class‑level lock for thread‑safe initialization.
    """

    _instance: Optional["ConfigManager"] = None
    _lock: threading.Lock = threading.Lock()

    # Declare attribute types at class level (important)
    _config: Optional[HypotestConfig]
    _initialized: bool

    def __new__(cls) -> "ConfigManager":
        """
        Create or return the singleton instance.

        Returns
        -------
        ConfigManager
            The singleton instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._config = None
                cls._instance._initialized = False
        return cls._instance

    def initialize(self, config: HypotestConfig) -> None:
        """
        Set the global configuration.

        Parameters
        ----------
        config : HypotestConfig
            The configuration to use globally.

        Raises
        ------
        ValueError
            If the configuration fails its own validation (e.g., missing API key).
        """
        # __post_init__ will run automatically, but we also call validation
        # by simply constructing the config (already validated).
        self._config = config
        self._initialized = True

    def get_config(self) -> HypotestConfig:
        """
        Retrieve the current global configuration.

        If the configuration has never been explicitly initialized, a default
        configuration (with `enable_llm_interpretation=False`) is created and
        returned. Subsequent calls will return the same default instance.

        Returns
        -------
        HypotestConfig
            The active configuration.
        """
        if self._config is None:
            # Lazy initialization with defaults
            self._config = HypotestConfig()
            self._initialized = False  # because it's an auto‑generated default
        return self._config

    def is_initialized(self) -> bool:
        """
        Check whether the configuration has been explicitly initialized by the user.

        Returns
        -------
        bool
            True if `initialize()` has been called with a user‑provided config,
            False if the configuration is still the default (lazy) one.
        """
        return self._initialized


# Public global instance
_config_manager = ConfigManager()


def initialize(config: HypotestConfig) -> None:
    """
    Initialize the global hypotest configuration.

    This is a convenience wrapper around `ConfigManager.initialize()`.

    Parameters
    ----------
    config : HypotestConfig
        The configuration to use globally.

    Examples
    --------
    >>> from hypotest.config import HypotestConfig, initialize
    >>> config = HypotestConfig(enable_llm_interpretation=True, llm_api_key="sk‑...")
    >>> initialize(config)
    """
    _config_manager.initialize(config)


def get_config() -> HypotestConfig:
    """
    Get the current global configuration.

    Returns
    -------
    HypotestConfig
        The active configuration.

    Examples
    --------
    >>> from hypotest.config import get_config
    >>> config = get_config()
    >>> config.enable_llm_interpretation
    False
    """
    return _config_manager.get_config()


def is_initialized() -> bool:
    """
    Check if the configuration has been explicitly initialized.

    Returns
    -------
    bool
        True if `initialize()` has been called, False otherwise.

    Examples
    --------
    >>> from hypotest.config import is_initialized
    >>> is_initialized()
    False
    """
    return _config_manager.is_initialized()