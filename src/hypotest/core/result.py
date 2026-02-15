"""
Result containers for hypothesis testing.

This module defines the primary data structures for storing and representing
results of statistical tests and complete analyses.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import json
import datetime


@dataclass
class TestResult:
    """
    Container for an individual statistical test result.

    Attributes
    ----------
    test_name : str
        Name of the test performed (e.g., "Independent t‑test").
    feature : str
        Name of the feature (variable) being tested, or a description of the
        comparison (e.g., "group_A vs group_B").
    statistic : float
        Test statistic value (e.g., t‑value, F‑value, U‑value).
    p_value : float
        P‑value of the test.
    effect_size : Optional[Any]
        Effect size measure (e.g., Cohen's d, η²). Placeholder until effect size
        module is implemented.
    assumptions : List[Any]
        List of assumption check results. Each element may be a dictionary,
        a dataclass, or a simple boolean indicating whether the assumption
        was met. Placeholder until assumption module is implemented.
    metadata : Dict[str, Any]
        Additional test‑specific metadata (degrees of freedom, sample sizes,
        confidence intervals, etc.).

    Examples
    --------
    >>> result = TestResult(
    ...     test_name="Independent t‑test",
    ...     feature="age",
    ...     statistic=2.345,
    ...     p_value=0.019,
    ...     effect_size={"cohens_d": 0.45},
    ...     assumptions=[{"normality": True, "p": 0.23}],
    ...     metadata={"df": 58, "n1": 30, "n2": 30}
    ... )
    >>> result.is_significant()
    True
    >>> result.is_significant(alpha=0.01)
    False
    """

    test_name: str
    feature: str
    statistic: float
    p_value: float
    effect_size: Optional[Any] = None
    assumptions: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """
        Determine if the test result is statistically significant at given alpha.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level (type I error rate).

        Returns
        -------
        bool
            True if p‑value ≤ alpha, otherwise False.

        Raises
        ------
        ValueError
            If alpha is not between 0 and 1.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        return self.p_value <= alpha

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test result to a serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization.
            Complex fields (effect_size, assumptions) are included as‑is;
            they must be JSON‑serializable by the caller.

        Notes
        -----
        This method uses `dataclasses.asdict` which recursively converts
        nested dataclasses. Non‑dataclass objects (e.g., numpy arrays) may
        need custom handling before serialization.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """
        Return a concise, human‑readable representation.
        """
        sig = "significant" if self.is_significant() else "non‑significant"
        return (
            f"TestResult(test='{self.test_name}', feature='{self.feature}', "
            f"statistic={self.statistic:.4f}, p={self.p_value:.4f}, {sig})"
        )

    def explain(self) -> Optional[str]:
        """
        Generate an LLM‑based interpretation of this test result.

        This method is optional and requires the library to be configured with
        `enable_llm_interpretation=True` and a valid `llm_api_key`.

        Returns
        -------
        Optional[str]
            A natural‑language interpretation of the test result, or None if
            LLM interpretation is disabled.

        Raises
        ------
        RuntimeError
            If LLM interpretation is enabled but no API key is configured,
            or if the LLM service fails.
        """
        # Lazy import to avoid circular dependencies and ensure no LLM
        # dependencies are loaded unless explicitly needed.
        from ..config.manager import get_config
        config = get_config()

        if not config.enable_llm_interpretation:
            return None

        if not config.llm_api_key:
            raise RuntimeError(
                "LLM interpretation is enabled but no API key is configured. "
                "Please set `llm_api_key` via `hypotest.configure()`."
            )

        # Lazy import of the interpreter to avoid pulling in LLM dependencies
        # when they are not used.
        from ..llm.interpretor import interpret_test_result
        explanation =  interpret_test_result(
            result=self,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
        return explanation.encode("utf-8", errors="replace").decode("utf-8")


@dataclass
class AnalysisResult:
    """
    Container for a complete analysis result.

    Aggregates exploratory data analysis (EDA) results, one or more test
    results, and provides recommendations, warnings, and an optional
    natural‑language interpretation.

    Attributes
    ----------
    eda_results : Optional[Any]
        Results of exploratory data analysis (descriptive statistics,
        distribution summaries, missingness reports, etc.).
    test_results : List[TestResult]
        List of test results, one per hypothesis tested.
    recommendations : List[str]
        Actionable recommendations based on the analysis (e.g., "Consider
        a non‑parametric alternative because normality assumption is violated").
    warnings : List[str]
        Warnings about potential issues (e.g., "Small sample size may reduce
        power", "Missing data may bias results").
    interpretation : Optional[str]
        Human‑readable interpretation of the overall analysis. May be
        generated by an LLM or a rule‑based formatter.

    Examples
    --------
    >>> analysis = AnalysisResult(
    ...     eda_results={"mean": 5.2, "std": 1.8},
    ...     test_results=[test_result],
    ...     recommendations=["Increase sample size for higher power"],
    ...     warnings=["Data is slightly skewed"],
    ...     interpretation="The test indicates a statistically significant..."
    ... )
    >>> print(analysis.summary())
    Analysis completed with 1 test(s).
    1 significant, 0 non‑significant.
    Recommendations: 1, Warnings: 1.
    """

    eda_results: Optional[Any] = None
    test_results: List[TestResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    interpretation: Optional[str] = None

    def summary(self) -> str:
        """
        Generate a clean human‑readable summary of the analysis.

        Returns
        -------
        str
            Summary string containing counts of tests, significance breakdown,
            and counts of recommendations and warnings.

        Examples
        --------
        >>> analysis.summary()
        "Analysis completed with 3 test(s).\n2 significant, 1 non‑significant.\nRecommendations: 2, Warnings: 1."
        """
        n_tests = len(self.test_results)
        n_sig = sum(1 for tr in self.test_results if tr.is_significant())
        n_nonsig = n_tests - n_sig
        lines = [
            f"Analysis completed with {n_tests} test(s).",
            f"{n_sig} significant, {n_nonsig} non‑significant.",
            f"Recommendations: {len(self.recommendations)}, Warnings: {len(self.warnings)}."
        ]
        if self.interpretation:
            lines.append(f"Interpretation: {self.interpretation[:100]}...")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the analysis result to a serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization.
            The `test_results` field is converted using each TestResult's
            `to_dict` method. Other fields are included as‑is.

        Notes
        -----
        For safe JSON serialization, ensure that `eda_results` and any
        nested objects are JSON‑serializable (e.g., convert numpy arrays to
        lists, datetime objects to ISO strings). This method does not perform
        automatic conversion; the caller is responsible for providing
        serializable data.
        """
        return {
            "eda_results": self.eda_results,
            "test_results": [tr.to_dict() for tr in self.test_results],
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "interpretation": self.interpretation,
            "_metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "hypotest_version": "0.1.0"
            }
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serialize the analysis result to a JSON string.

        Parameters
        ----------
        indent : Optional[int], default 2
            Indentation level for pretty‑printing. Pass None for compact output.

        Returns
        -------
        str
            JSON string representation.

        Raises
        ------
        TypeError
            If any field is not JSON‑serializable.
        """
        return json.dumps(self.to_dict(), indent=indent, default=self._json_default)

    @staticmethod
    def _json_default(obj):
        """
        Default JSON serializer for non‑standard types.

        Handles datetime objects and numpy arrays (if numpy is available).
        """
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Attempt to convert numpy types (optional)
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Fallback: raise TypeError
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def __repr__(self) -> str:
        """
        Return a concise representation of the analysis result.
        """
        n_tests = len(self.test_results)
        n_sig = sum(1 for tr in self.test_results if tr.is_significant())
        return (
            f"AnalysisResult(tests={n_tests}, significant={n_sig}, "
            f"recommendations={len(self.recommendations)}, "
            f"warnings={len(self.warnings)})"
        )