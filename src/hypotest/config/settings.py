"""
Configuration dataclasses for hypotest.

This module defines the core configuration structure used throughout the library.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HypotestConfig:
    """
    Immutable configuration for the hypotest library.

    Attributes
    ----------
    enable_llm_interpretation : bool, default False
        Whether to enable LLM‑based interpretation of test results.
        If False, calls to `TestResult.explain()` will return None.
    llm_api_key : Optional[str], default None
        API key for the LLM service (e.g., OpenAI, Anthropic, or compatible).
        Required when `enable_llm_interpretation` is True.
    llm_base_url : Optional[str], default None
        Base URL for the LLM API. If None, the default OpenAI‑compatible URL is used.
    llm_model : str, default "gpt-4o-mini"
        Model identifier to use for interpretation (e.g., "gpt-4o", "claude-3-haiku").
    """

    enable_llm_interpretation: bool = False
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises
        ------
        ValueError
            If `enable_llm_interpretation` is True but `llm_api_key` is missing.
        """
        if self.enable_llm_interpretation and not self.llm_api_key:
            raise ValueError(
                "LLM interpretation is enabled but no API key was provided. "
                "Set `llm_api_key` or disable `enable_llm_interpretation`."
            )