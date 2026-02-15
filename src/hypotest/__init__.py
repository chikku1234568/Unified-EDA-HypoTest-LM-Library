"""
Hypotest - A Python library for hypothesis testing with automated assumptions.
"""

__version__ = "0.1.0"
__author__ = "Hypotest Contributors"

from .core import *
from .analysis import *
from .assumptions import *
from .effect_size import *
from .reporting import *
from .llm import *
from .config import *
from .utils import *

from typing import Optional
from .config import HypotestConfig, initialize as _initialize


def configure(
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    enable_llm_interpretation: bool = False,
) -> None:
    """
    Configure the hypotest library globally.

    This function initializes the library's global configuration with the provided
    settings. If called multiple times, the last call determines the active
    configuration.

    Parameters
    ----------
    llm_api_key : Optional[str], default None
        API key for the LLM service (e.g., OpenAI, Anthropic, or compatible).
        Required when `enable_llm_interpretation` is True.
    llm_base_url : Optional[str], default None
        Base URL for the LLM API. If None, the default OpenAI‑compatible URL is used.
    llm_model : str, default "gpt-4o-mini"
        Model identifier to use for interpretation (e.g., "gpt-4o", "claude-3-haiku").
    enable_llm_interpretation : bool, default False
        Whether to enable LLM‑based interpretation of test results.
        If False, calls to `TestResult.explain()` will return None.

    Examples
    --------
    >>> import hypotest
    >>> hypotest.configure(
    ...     llm_api_key="sk‑...",
    ...     enable_llm_interpretation=True
    ... )
    """
    config = HypotestConfig(
        enable_llm_interpretation=enable_llm_interpretation,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )
    _initialize(config)


from .info import info

# Ensure the function is exposed
__all__ = [
    "configure",
    "info",
    "__version__",
    "__author__",
]