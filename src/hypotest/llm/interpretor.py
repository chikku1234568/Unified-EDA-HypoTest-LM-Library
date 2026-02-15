# hypotest/llm/interpreter.py

from typing import Optional
from .client import OpenAICompatibleClient
from hypotest.core.result import TestResult


SYSTEM_PROMPT = """
You are a statistical expert.

Your job is to interpret statistical test results produced by a verified statistical engine.

DO NOT fabricate numbers.
DO NOT modify numbers.
ONLY interpret the provided results.

Explain:

- whether the result is statistically significant
- what the effect size means
- whether assumptions are violated
- practical interpretation

Be precise and scientifically accurate.
"""


def interpret_test_result(
    result: TestResult,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    assumptions_text = "\n".join(
        f"- {a.assumption_name}: {'satisfied' if a.passed else 'violated'}"
        for a in result.assumptions
    )

    user_prompt = f"""
Test: {result.test_name}

Statistic: {result.statistic}

P-value: {result.p_value}

Effect size: {result.effect_size}

Assumptions:
{assumptions_text}
"""

    return client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
    )
