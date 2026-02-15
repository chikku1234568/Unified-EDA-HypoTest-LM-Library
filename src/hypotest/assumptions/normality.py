"""
Normality assumption checker using the Shapiro-Wilk test.

This module provides the NormalityChecker class, which implements the
AssumptionChecker interface for checking whether a univariate sample
comes from a normally distributed population.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.stats

from .base import AssumptionChecker, AssumptionResult


class NormalityChecker(AssumptionChecker):
    """
    Checker for the normality assumption using the Shapiro-Wilk test.

    The Shapiro-Wilk test is a widely used test of normality that is
    appropriate for small to moderately sized samples (up to ~5000 observations).
    It tests the null hypothesis that the data were drawn from a normal
    distribution.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Examples
    --------
    >>> checker = NormalityChecker()
    >>> import pandas as pd
    >>> data = pd.Series([1.2, 2.3, 1.8, 2.1, 1.9])
    >>> result = checker.check(data, alpha=0.05)
    >>> result.assumption_name
    'normality'
    >>> isinstance(result.passed, bool)
    True
    >>> result.statistic is not None
    True
    >>> result.p_value is not None
    True
    """

    def get_name(self) -> str:
        """
        Return the canonical name of the assumption checked by this class.

        Returns
        -------
        str
            The string ``"normality"``.
        """
        return "normality"

    def check(self, data: pd.Series, alpha: float = 0.05, **kwargs: Any) -> AssumptionResult:
        """
        Perform a Shapiro‑Wilk test for normality.

        Missing values are automatically removed before testing. If the
        sample size after removing missing values is less than 3, the
        assumption is considered **not satisfied** and a descriptive
        interpretation is returned.

        Parameters
        ----------
        data : pd.Series
            A 1‑D array‑like object containing the sample to be tested.
            Can be a pandas Series, a numpy array, or any sequence that
            pandas can interpret as a Series.
        alpha : float, default 0.05
            Significance level for the test. A p‑value greater than or
            equal to ``alpha`` leads to ``passed = True`` (i.e., the
            normality assumption is not rejected).
        **kwargs : Any
            Additional keyword arguments are ignored (present for future
            extensibility).

        Returns
        -------
        AssumptionResult
            A dataclass containing:

            - ``assumption_name`` : ``"normality"``
            - ``passed`` : bool indicating whether the assumption is
              considered satisfied (p ≥ alpha)
            - ``statistic`` : float, the Shapiro‑Wilk W statistic
            - ``p_value`` : float, the p‑value of the test
            - ``interpretation`` : str, a human‑readable summary of the
              result
            - ``recommendation`` : str or None, a suggestion for the user
              when the assumption is violated

        Raises
        ------
        TypeError
            If ``data`` cannot be converted to a pandas Series.
        ValueError
            If ``alpha`` is not between 0 and 1, or if the data contain
            no finite values after removing missing values.

        Notes
        -----
        The Shapiro‑Wilk test is known to be sensitive to outliers and
        can be overly powerful for large samples (i.e., it may detect
        trivial deviations from normality that are not practically
        important). For sample sizes above 5000, consider using the
        Anderson‑Darling or Kolmogorov‑Smirnov test instead.

        The test is only valid for sample sizes between 3 and 5000.
        For smaller samples the method returns a ``passed = False``
        result with an appropriate interpretation.
        """
        # Validate alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # Convert input to a pandas Series (makes dropna and size checking easy)
        if not isinstance(data, pd.Series):
            try:
                data = pd.Series(data)
            except Exception as e:
                raise TypeError(
                    "data must be convertible to a pandas Series"
                ) from e

        # Remove missing values safely
        clean_data = data.dropna()
        n = len(clean_data)

        # Insufficient sample size
        if n < 3:
            interpretation = (
                f"Sample size after removing missing values is {n}, "
                "which is insufficient for the Shapiro‑Wilk test "
                "(requires at least 3 observations)."
            )
            return AssumptionResult(
                assumption_name=self.get_name(),
                passed=False,
                statistic=None,
                p_value=None,
                interpretation=interpretation,
                recommendation=(
                    "Collect more data or consider a non‑parametric test "
                    "that does not require normality."
                ),
            )

        # Perform Shapiro‑Wilk test
        # scipy.stats.shapiro raises a warning if n > 5000; we'll let it warn.
        try:
            statistic, p_value = scipy.stats.shapiro(clean_data.values)
        except Exception as e:
            # Fallback for unexpected errors (e.g., all identical values?)
            raise RuntimeError(
                f"Shapiro‑Wilk test failed with error: {e}"
            ) from e

        # Determine whether the assumption is satisfied
        passed = p_value >= alpha

        # Build interpretation
        if passed:
            interpretation = (
                f"Data appear normally distributed (p = {p_value:.4f} ≥ alpha = {alpha})."
            )
        else:
            interpretation = (
                f"Data do not appear normally distributed (p = {p_value:.4f} < alpha = {alpha})."
            )

        # Recommendation only when assumption is violated
        recommendation = None
        if not passed:
            recommendation = (
                "Consider using non‑parametric tests such as "
                "Mann‑Whitney U or Kruskal‑Wallis."
            )

        return AssumptionResult(
            assumption_name=self.get_name(),
            passed=passed,
            statistic=statistic,
            p_value=p_value,
            interpretation=interpretation,
            recommendation=recommendation,
        )

    def __repr__(self) -> str:
        """Return a simple representation of the checker."""
        return f"NormalityChecker(name='{self.get_name()}')"