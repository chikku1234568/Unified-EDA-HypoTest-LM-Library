"""
Library self-description and configuration status.

This module provides a function that returns a detailed, human‑readable overview
of the hypotest library, its capabilities, and its current configuration.
"""

import textwrap
from typing import List
from .config.manager import get_config
from .tests.registry import TestRegistry


def info() -> str:
    """
    Return a professional description of hypotest and its configuration.

    The description includes:

    * what hypotest does
    * statistical engine capabilities
    * assumption checking
    * registry system
    * optional LLM interpretation
    * deterministic statistical guarantees
    * current configuration status (LLM enabled, model, API key, etc.)

    Returns
    -------
    str
        A formatted, multi-line string suitable for printing to the console
        or logging.

    Examples
    --------
    >>> import hypotest
    >>> print(hypotest.info())
    ========================================
    hypotest - Hypothesis Testing Library
    ========================================
    ...
    """
    config = get_config()

    # Gather registered test names
    try:
        registered_names: List[str] = TestRegistry.get_registered_names()
        registered_count = len(registered_names)
        registered_summary = (
            f"{registered_count} registered test(s): "
            + ", ".join(registered_names[:5])
            + (" ..." if registered_count > 5 else "")
        )
    except Exception:
        registered_summary = "(registry not available)"

    # Build the description
    lines = [
    "=" * 60,
    "hypotest - Hypothesis Testing Library",
    "=" * 60,
    "",
    "OVERVIEW",
    "-------",
    "hypotest is a Python library for performing automated hypothesis testing",
    "with built-in assumption checking, effect size computation, and optional",
    "LLM-based interpretation. It is designed for data scientists, researchers,",
    "and analysts who need reproducible, assumption-aware statistical workflows.",
    "",
    "CORE CAPABILITIES",
    "----------------",
    "- Statistical engine: parametric and non-parametric tests with exact",
    "  p-values, test statistics, and effect sizes.",
    "- Assumption checking: automated verification of normality, variance",
    "  homogeneity, independence, and other test-specific assumptions.",
    "- Registry system: plug-in architecture for adding custom statistical",
    "  tests via the @register_test decorator.",
    "- Deterministic guarantees: all statistical computations are fully",
    "  deterministic and reproducible across runs.",
    "",
    "DATA HANDLING",
    "------------",
    "- Uses a Dataset abstraction layer to safely wrap pandas DataFrames.",
    "- Automatically handles missing values in relevant variables.",
    "- Validates column existence and sample size before executing tests.",
    "",
    "AUTOMATIC VALIDATION",
    "-------------------",
    "- Automatically checks statistical assumptions before test execution.",
    "- Provides structured AssumptionResult objects with interpretation",
    "  and recommendations.",
    "- Prevents invalid statistical tests from running on unsuitable data.",
    "",
    "OPTIONAL LLM INTERPRETATION",
    "--------------------------",
    "- LLM-based interpretation can be enabled to generate natural-language",
    "  explanations of test results.",
    "- Requires an API key and a compatible LLM endpoint (OpenAI-compatible).",
    "- Statistical computations remain deterministic and independent of LLM.",
    "",
    "CURRENT CONFIGURATION",
    "---------------------",
    f"- LLM interpretation enabled: {'Yes' if config.enable_llm_interpretation else 'No'}",
    f"- LLM model: {config.llm_model}",
    f"- API key configured: {'Yes' if config.llm_api_key else 'No'}",
    f"- Base URL: {config.llm_base_url if config.llm_base_url else '(default)'}",
    f"- Registry status: {registered_summary}",
    "",
    "VERSION",
    "-------",
    "hypotest 0.1.0 - https://github.com/chikku1234568/Unified-EDA-HypoTest-LM-Library",
    "",
    "USAGE EXAMPLE",
    "-------------",
    "  import pandas as pd",
    "  from hypotest.core.dataset import Dataset",
    "",
    "  df = pd.DataFrame({...})",
    "  dataset = Dataset(df)",
    "",
    "  test = hypotest.TTest()",
    "  result = test.execute(dataset, target='value', features=['group'])",
    "",
    "  print(result)",
    "  print(result.explain())",
    "",
    "=" * 60,
]

    # Dedent and join
    return "\n".join(lines)