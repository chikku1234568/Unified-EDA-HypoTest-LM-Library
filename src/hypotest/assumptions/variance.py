"""
Variance homogeneity (homoscedasticity) assumption checker using Levene's test.

This module provides the VarianceChecker class, which implements the
AssumptionChecker interface for checking whether multiple groups have equal
variances (homoscedasticity). The test is performed via Levene's test, which
is robust to departures from normality.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import scipy.stats

from .base import AssumptionChecker, AssumptionResult


class VarianceChecker(AssumptionChecker):
    """
    Checker for the variance homogeneity assumption using Levene's test.

    Levene's test assesses the null hypothesis that all input groups have equal
    variances. It is less sensitive to departures from normality than Bartlett's
    test and is suitable for comparing two or more groups.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Examples
    --------
    >>> checker = VarianceChecker()
    >>> import pandas as pd
    >>> group1 = pd.Series([1.2, 2.3, 1.8, 2.1, 1.9])
    >>> group2 = pd.Series([2.8, 3.2, 3.0, 2.9, 3.1])
    >>> result = checker.check([group1, group2], alpha=0.05)
    >>> result.assumption_name
    'variance_homogeneity'
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
            The string ``"variance_homogeneity"``.
        """
        return "variance_homogeneity"

    def check(
        self,
        groups: List[pd.Series],
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> AssumptionResult:
        """
        Perform Levene's test for equality of variances across groups.

        Missing values are automatically removed from each group before testing.
        The method validates that at least two groups are provided and each group
        contains at least two observations after cleaning. If these conditions
        are not met, the assumption is considered **not satisfied** and a
        descriptive interpretation is returned.

        Parameters
        ----------
        groups : List[pd.Series]
            A list of pandas Series, each representing a distinct group.
            The Series can contain missing values (NaN), which will be dropped
            per group.
        alpha : float, default 0.05
            Significance level for the test. A p‑value greater than or equal to
            ``alpha`` leads to ``passed = True`` (i.e., the homogeneity
            assumption is not rejected).
        **kwargs : Any
            Additional keyword arguments are ignored (present for future
            extensibility). Levene's test supports a ``center`` parameter
            (``'median'``, ``'mean'``, ``'trimmed'``); the default ``'median'``
            is used.

        Returns
        -------
        AssumptionResult
            A dataclass containing:

            - ``assumption_name`` : ``"variance_homogeneity"``
            - ``passed`` : bool indicating whether the assumption is
              considered satisfied (p ≥ alpha)
            - ``statistic`` : float, Levene's test statistic
            - ``p_value`` : float, the p‑value of the test
            - ``interpretation`` : str, a human‑readable summary of the result
            - ``recommendation`` : str or None, a suggestion for the user when
              the assumption is violated

        Raises
        ------
        TypeError
            If ``groups`` is not a list or its elements cannot be converted to
            pandas Series.
        ValueError
            If ``alpha`` is not between 0 and 1, or if the data contain
            no finite values after removing missing values.
        RuntimeError
            If Levene's test fails unexpectedly (e.g., due to numerical issues).

        Notes
        -----
        Levene's test is robust to moderate deviations from normality, making it
        a safer choice than Bartlett's test when normality cannot be assumed.
        However, the test still requires that each group has at least two
        observations after cleaning. For very small sample sizes, the test may
        have low power.

        The test is performed with ``center='median'`` (the default in
        ``scipy.stats.levene``), which provides the best robustness against
        non‑normal distributions.
        """
        # Validate alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # Ensure groups is a list
        if not isinstance(groups, list):
            raise TypeError("groups must be a list of pandas Series")

        if len(groups) < 2:
            interpretation = (
                f"At least two groups are required, got {len(groups)}."
            )
            return AssumptionResult(
                assumption_name=self.get_name(),
                passed=False,
                statistic=None,
                p_value=None,
                interpretation=interpretation,
                recommendation=(
                    "Provide at least two groups to test variance homogeneity."
                ),
            )

        # Convert each element to a pandas Series and drop missing values
        clean_groups = []
        for i, g in enumerate(groups):
            if not isinstance(g, pd.Series):
                try:
                    g = pd.Series(g)
                except Exception as e:
                    raise TypeError(
                        f"group {i} must be convertible to a pandas Series"
                    ) from e
            cleaned = g.dropna()
            if len(cleaned) < 2:
                interpretation = (
                    f"Group {i} has only {len(cleaned)} observation(s) after "
                    "removing missing values; at least 2 are required."
                )
                return AssumptionResult(
                    assumption_name=self.get_name(),
                    passed=False,
                    statistic=None,
                    p_value=None,
                    interpretation=interpretation,
                    recommendation=(
                        "Ensure each group contains at least two non‑missing "
                        "observations."
                    ),
                )
            clean_groups.append(cleaned.values)

        # At this point we have at least two groups, each with ≥2 observations
        try:
            # Use Levene's test with default center='median'
            statistic, p_value = scipy.stats.levene(*clean_groups)
        except Exception as e:
            # Catch any numerical or input errors from scipy
            raise RuntimeError(
                f"Levene's test failed with error: {e}"
            ) from e

        # Determine whether the assumption is satisfied
        passed = p_value >= alpha

        # Build interpretation
        if passed:
            interpretation = (
                f"Variances appear equal across groups (p = {p_value:.4f} ≥ alpha = {alpha})."
            )
        else:
            interpretation = (
                f"Variances differ significantly across groups (p = {p_value:.4f} < alpha = {alpha})."
            )

        # Recommendation only when assumption is violated
        recommendation = None
        if not passed:
            recommendation = (
                "Consider using Welch’s t‑test or non‑parametric tests "
                "that do not assume equal variances."
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
        return f"VarianceChecker(name='{self.get_name()}')"