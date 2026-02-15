# hypotest/llm/schemas.py

from dataclasses import dataclass
from typing import List


@dataclass
class InterpretationRequest:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    assumptions: List[str]


@dataclass
class InterpretationResponse:
    explanation: str
