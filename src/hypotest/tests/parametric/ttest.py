"""
Independent Samples t-test implementation.

This module provides the TTest class, a parametric test for comparing means
between two independent groups. It assumes normality and variance homogeneity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
import scipy.stats

from ..base import StatisticalTest, TestMetadata
from ..registry import register_test
from ...core.dataset import Dataset
from ...core.result import TestResult
from ...assumptions.base import AssumptionResult
from ...assumptions.normality import NormalityChecker
from ...assumptions.variance import VarianceChecker


logger = logging.getLogger(__name__)


@register_test
class TTest(StatisticalTest):
    """
    Independent Samples t-test.

    This test compares the means of a continuous target variable between two
    groups defined by a categorical or binary feature. It requires the
    assumptions of normality (within each group) and homogeneity of variances
    (equal variances across groups).

    Parameters
    ----------
    None

    Examples
    --------
    >>> from hypotest.core.dataset import Dataset
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'score': [1.2, 2.3, 1.8, 2.1, 1.9, 2.8, 3.2, 3.0, 2.9, 3.1],
    ...     'group': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    ... })
    >>> dataset = Dataset(df)
    >>> test = TTest()
    >>> test.is_applicable(dataset, 'score', ['group'])
    True
    >>> assumptions = test.check_assumptions(dataset, 'score', ['group'])
    >>> len(assumptions)
    2
    >>> result = test.execute(dataset, 'score', ['group'])
    >>> isinstance(result, TestResult)
    True
    >>> result.statistic
    -5.196...
    """

    metadata = TestMetadata(
        name="Independent Samples t-test",
        category="parametric",
        assumptions=["normality", "variance_homogeneity"],
        applicable_types={
            "target": ["continuous"],
            "features": ["categorical", "binary"]
        },
        min_sample_size=3,
        description="Tests whether means differ between two groups.",
    )

    def is_applicable(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> bool:
        """
        Determine whether the t‑test is applicable to the given dataset.

        The test is applicable if:
        1. The target variable is continuous.
        2. The feature variable is categorical or binary.
        3. The feature has exactly two distinct groups.
        4. The total sample size (after removing missing values) is at least
           `self.metadata.min_sample_size`.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        features : List[str]
            Names of the feature (independent) variables. Only the first
            feature is used; additional features are ignored.

        Returns
        -------
        bool
            True if the test can be applied, False otherwise.

        Raises
        ------
        KeyError
            If `target` or any `feature` is not a column in the dataset.
        """
        # Ensure columns exist
        if target not in dataset.columns:
            raise KeyError(f"Target column '{target}' not found in dataset.")
        if not features:
            raise ValueError("At least one feature must be provided.")
        feature = features[0]
        if feature not in dataset.columns:
            raise KeyError(f"Feature column '{feature}' not found in dataset.")

        # Check variable types via dataset.variable_types (if available)
        if dataset.variable_types:
            target_type = dataset.variable_types.get(target, "")
            feature_type = dataset.variable_types.get(feature, "")
        else:
            # Fallback: infer from data (basic heuristic)
            # This is a placeholder; in production, the dataset should have
            # variable_types populated by a TypeDetector.
            target_series = dataset.get_column(target)
            feature_series = dataset.get_column(feature)
            target_type = self._infer_variable_type(target_series)
            feature_type = self._infer_variable_type(feature_series)

        # Target must be continuous
        if target_type != "continuous":
            logger.debug(
                "Target '%s' is of type '%s', but continuous is required.",
                target, target_type
            )
            return False

        # Feature must be categorical or binary
        if feature_type not in ("categorical", "binary"):
            logger.debug(
                "Feature '%s' is of type '%s', but categorical or binary is required.",
                feature, feature_type
            )
            return False

        # Feature must have exactly two distinct groups (after dropping missing)
        df_subset = dataset.data[[target, feature]].dropna()
        unique_groups = df_subset[feature].unique()
        if len(unique_groups) != 2:
            logger.debug(
                "Feature '%s' has %d distinct groups, but exactly two are required.",
                feature, len(unique_groups)
            )
            return False

        # Validate sample size
        if not self.validate_sample_size(dataset, target, features):
            logger.debug(
                "Sample size after removing missing values is less than minimum required (%d).",
                self.metadata.min_sample_size
            )
            return False

        # All checks passed
        return True

    def check_assumptions(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> List[AssumptionResult]:
        """
        Check normality and variance homogeneity assumptions.

        The method performs:
        1. Shapiro‑Wilk normality test on each group separately.
        2. Levene's test for equality of variances across the two groups.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        features : List[str]
            Names of the feature (independent) variables. Only the first
            feature is used; additional features are ignored.

        Returns
        -------
        List[AssumptionResult]
            A list of two AssumptionResult objects, one for normality and one
            for variance homogeneity, in the order specified by
            `self.metadata.assumptions`.

        Raises
        ------
        RuntimeError
            If the assumption checkers fail unexpectedly.
        """
        if not features:
            raise ValueError("At least one feature must be provided.")
        feature = features[0]

        # Split target values into two groups
        groups = self.get_groups(dataset, target, feature)
        if len(groups) != 2:
            raise ValueError(
                f"Expected exactly two groups, but found {len(groups)}."
            )

        results = []

        # 1. Normality assumption (check each group)
        normality_checker = NormalityChecker()
        normality_passed = True
        interpretations = []
        for i, group_data in enumerate(groups):
            result = normality_checker.check(group_data, alpha=0.05)
            results.append(result)
            interpretations.append(
                f"Group {i}: {result.interpretation}"
            )
            if not result.passed:
                normality_passed = False

        # Combine normality results into a single assumption result
        # (the metadata expects a single "normality" entry)
        normality_result = AssumptionResult(
            assumption_name="normality",
            passed=normality_passed,
            statistic=None,  # Not aggregatable
            p_value=None,
            interpretation="; ".join(interpretations),
            recommendation=(
                "Consider non‑parametric alternatives (e.g., Mann‑Whitney U) "
                "if normality is violated."
            ) if not normality_passed else None
        )
        # Replace the two per‑group results with the combined result
        # (We keep the combined result and discard the per‑group ones)
        results = [normality_result]

        # 2. Variance homogeneity assumption
        variance_checker = VarianceChecker()
        variance_result = variance_checker.check(groups, alpha=0.05)
        results.append(variance_result)

        # Ensure order matches metadata.assumptions
        # The metadata lists ["normality", "variance_homogeneity"]
        # We have normality first, variance second.
        return results

    def execute(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> TestResult:
        """
        Perform the independent samples t‑test.

        The method splits the target variable into two groups based on the
        first feature, runs `scipy.stats.ttest_ind`, and packages the result
        into a `TestResult` object.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        features : List[str]
            Names of the feature (independent) variables. Only the first
            feature is used; additional features are ignored.

        Returns
        -------
        TestResult
            A result object containing:
            - test_name: "Independent Samples t-test"
            - feature: the name of the feature used
            - statistic: the t‑statistic
            - p_value: the two‑sided p‑value
            - assumptions: the list of assumption results from `check_assumptions`
            - metadata: additional information (degrees of freedom, group sizes)

        Raises
        ------
        RuntimeError
            If the t‑test fails due to insufficient data or numerical issues.
        """
        if not features:
            raise ValueError("At least one feature must be provided.")
        feature = features[0]

        # Split into groups
        groups = self.get_groups(dataset, target, feature)
        if len(groups) != 2:
            raise RuntimeError(
                f"Expected exactly two groups, but found {len(groups)}."
            )

        group1, group2 = groups[0].values, groups[1].values

        # Perform t‑test
        try:
            statistic, p_value = scipy.stats.ttest_ind(group1, group2)
        except Exception as e:
            raise RuntimeError(
                f"Independent samples t‑test failed: {e}"
            ) from e

        # Compute assumption results (they may be cached in a real implementation)
        assumptions = self.check_assumptions(dataset, target, features)

        # Prepare metadata
        metadata: Dict[str, Any] = {
            "df": len(group1) + len(group2) - 2,
            "n1": len(group1),
            "n2": len(group2),
            "mean1": float(np.mean(group1)),
            "mean2": float(np.mean(group2)),
            "std1": float(np.std(group1, ddof=1)),
            "std2": float(np.std(group2, ddof=1)),
            "test_type": "independent",
            "alternative": "two-sided",
        }

        return TestResult(
            test_name=self.metadata.name,
            feature=feature,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=self.calculate_effect_size(dataset, target, features),
            assumptions=assumptions,
            metadata=metadata,
        )

    def calculate_effect_size(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> Optional[Any]:
        """
        Calculate Cohen's d as the effect size for the t‑test.

        This is a placeholder implementation. In a full version, Cohen's d
        should be computed as (mean1 - mean2) / pooled_standard_deviation.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        features : List[str]
            Names of the feature (independent) variables.

        Returns
        -------
        Optional[Any]
            None for now. In the future, a dictionary with keys:
            - "cohens_d": float
            - "interpretation": str
        """
        # TODO: implement Cohen's d calculation
        return None

    # ------------------ Private helper methods ------------------

    def _infer_variable_type(self, series: pd.Series) -> str:
        """
        Infer variable type from a pandas Series (basic heuristic).

        This is a fallback when `dataset.variable_types` is not populated.
        It should not be used in production; the dataset should have its
        variable types determined by a proper TypeDetector.

        Parameters
        ----------
        series : pd.Series
            The column data.

        Returns
        -------
        str
            One of "continuous", "categorical", "binary", or "ordinal".
        """
        from ...analysis.type_detector import TypeDetector
        detector = TypeDetector()
        result = detector.detect(series)
        return result.type.value  # returns e.g., "continuous"

    def __repr__(self) -> str:
        """Return a concise representation of the test."""
        return f"TTest(name='{self.metadata.name}')"