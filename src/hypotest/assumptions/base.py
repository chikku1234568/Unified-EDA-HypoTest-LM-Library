"""
Base classes for statistical assumption checking.

This module defines the fundamental data structure for assumption results
(AssumptionResult) and the abstract interface for assumption checkers
(AssumptionChecker).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class AssumptionResult:
    """
    Container for the result of a statistical assumption check.

    Attributes
    ----------
    assumption_name : str
        Name of the assumption being checked (e.g., "normality", "homoscedasticity").
    passed : bool
        Whether the assumption is satisfied according to the chosen criterion.
    statistic : Optional[float]
        Test statistic value, if applicable (e.g., Shapiro‑Wilk W, Levene's F).
    p_value : Optional[float]
        P‑value of the test, if applicable.
    interpretation : str
        Human‑readable interpretation of the result, explaining what the
        assumption check means in context.
    recommendation : Optional[str]
        Optional recommendation for the user, e.g., "proceed with parametric test"
        or "consider a non‑parametric alternative".

    Examples
    --------
    >>> result = AssumptionResult(
    ...     assumption_name="normality",
    ...     passed=False,
    ...     statistic=0.923,
    ...     p_value=0.012,
    ...     interpretation="Data significantly deviate from normality (p < 0.05)",
    ...     recommendation="Use a non‑parametric test (e.g., Mann‑Whitney U)"
    ... )
    >>> result.is_valid()
    False
    >>> result.to_dict()["assumption_name"]
    'normality'
    """

    assumption_name: str
    passed: bool
    interpretation: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    recommendation: Optional[str] = None

    def is_valid(self, alpha: float = 0.05) -> bool:
        """
        Determine whether the assumption is considered valid.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level (ignored if the assumption does not rely on a
            p‑value). The default implementation simply returns `self.passed`;
            subclasses may override to incorporate `alpha` if desired.

        Returns
        -------
        bool
            True if the assumption is satisfied, False otherwise.

        Notes
        -----
        Many assumption checks are based on a hypothesis test where a p‑value
        above `alpha` indicates that the assumption is not violated. However,
        the `passed` field already encodes that decision (using the checker's
        internal alpha). This method provides a uniform interface for downstream
        code to query validity, optionally with a custom alpha.
        """
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization.
            Nested dataclasses are recursively converted via `dataclasses.asdict`.

        See Also
        --------
        dataclasses.asdict : Used internally for recursive conversion.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """
        Return a concise, human‑readable representation of the result.

        Examples
        --------
        >>> r = AssumptionResult("normality", False, "deviates from normality")
        >>> repr(r)
        "AssumptionResult(name='normality', passed=False, p=None)"
        """
        p_str = f"{self.p_value:.4f}" if self.p_value is not None else "None"
        return (
            f"AssumptionResult(name='{self.assumption_name}', "
            f"passed={self.passed}, p={p_str})"
        )


class AssumptionChecker(abc.ABC):
    """
    Abstract base class for statistical assumption checkers.

    Concrete subclasses must implement the `check` method, which performs the
    actual assumption test on the provided data, and the `get_name` method,
    which returns a string identifier for the assumption.

    The class is designed to be extensible: new assumption checks can be added
    by inheriting from `AssumptionChecker` and implementing the two abstract
    methods.

    Examples
    --------
    >>> class NormalityChecker(AssumptionChecker):
    ...     def get_name(self) -> str:
    ...         return "normality"
    ...     def check(self, data, **kwargs):
    ...         # perform Shapiro‑Wilk test
    ...         statistic, p = stats.shapiro(data)
    ...         passed = p > 0.05
    ...         return AssumptionResult(
    ...             assumption_name=self.get_name(),
    ...             passed=passed,
    ...             statistic=statistic,
    ...             p_value=p,
    ...             interpretation="Shapiro‑Wilk test for normality",
    ...             recommendation=None
    ...         )
    """

    @abc.abstractmethod
    def check(self, data: Any, **kwargs) -> AssumptionResult:
        """
        Perform the assumption check on the provided data.

        Parameters
        ----------
        data : Any
            The data to be checked. The exact type is defined by the concrete
            checker (e.g., a 1‑D array for normality, two groups for
            homoscedasticity).
        **kwargs
            Additional keyword arguments that may be needed by the specific
            checker (e.g., significance level, test variant).

        Returns
        -------
        AssumptionResult
            A dataclass containing the result of the assumption check.

        Raises
        ------
        ValueError
            If the data are malformed or insufficient for the check.
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the canonical name of the assumption checked by this class.

        Returns
        -------
        str
            A short, descriptive name (e.g., "normality", "independence",
            "homoscedasticity").
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return a simple representation of the checker.
        """
        return f"{self.__class__.__name__}(name='{self.get_name()}')"