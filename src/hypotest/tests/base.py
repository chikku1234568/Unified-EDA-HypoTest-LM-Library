"""
Base classes for statistical tests in the hypotest framework.

This module defines the core abstract base class for all statistical tests,
along with the metadata dataclass that describes test characteristics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..core.dataset import Dataset
from ..core.result import TestResult
from ..assumptions.base import AssumptionResult
from ..assumptions.normality import NormalityChecker
from ..assumptions.variance import VarianceChecker


@dataclass
class TestMetadata:
    """
    Metadata describing a statistical test.

    Attributes
    ----------
    name : str
        Canonical name of the test (e.g., "Independent t‑test").
    category : str
        Broad category of the test (e.g., "parametric", "non‑parametric",
        "difference‑of‑means", "correlation").
    assumptions : List[str]
        List of assumption identifiers that the test relies on.
        Examples: ["normality", "variance_homogeneity", "independence"].
    applicable_types : Dict[str, List[Any]]
        Mapping from variable role (e.g., "target", "feature") to allowed
        data types (e.g., ["continuous"], ["categorical", "binary"]).
        This is used by `is_applicable` to validate variable types.
    min_sample_size : int
        Minimum total sample size required for the test to be valid.
    description : Optional[str]
        Human‑readable description of the test, its purpose, and its
        interpretation.
    references : Optional[List[str]]
        Optional list of reference strings (e.g., academic papers, textbook
        sections, online resources) for the test.

    Examples
    --------
    >>> meta = TestMetadata(
    ...     name="Independent t‑test",
    ...     category="parametric",
    ...     assumptions=["normality", "variance_homogeneity"],
    ...     applicable_types={
    ...         "target": ["continuous"],
    ...         "feature": ["binary"]
    ...     },
    ...     min_sample_size=20,
    ...     description="Compares means between two independent groups.",
    ...     references=["Student (1908) Biometrika", "..."])
    >>> meta.name
    'Independent t‑test'
    """

    name: str
    category: str
    assumptions: List[str]
    applicable_types: Dict[str, List[Any]]
    min_sample_size: int
    description: Optional[str] = None
    references: Optional[List[str]] = None

    def __repr__(self) -> str:
        """Return a concise representation of the metadata."""
        return f"TestMetadata(name='{self.name}', category='{self.category}')"


class StatisticalTest(abc.ABC):
    """
    Abstract base class for all statistical tests.

    Concrete test classes must implement the four abstract methods
    `is_applicable`, `check_assumptions`, `execute`, and
    `calculate_effect_size`. They must also define a `metadata` attribute
    of type `TestMetadata`.

    The class provides several concrete helper methods that can be used
    across test implementations, such as `validate_sample_size` and
    `get_groups`.

    Attributes
    ----------
    metadata : TestMetadata
        Metadata describing the test's characteristics, assumptions, and
        applicability.

    Examples
    --------
    >>> class IndependentTTest(StatisticalTest):
    ...     metadata = TestMetadata(...)
    ...     def is_applicable(self, dataset, target, features):
    ...         # Check variable types and sample size
    ...         pass
    ...     def check_assumptions(self, dataset, target, features):
    ...         # Run normality and variance checks
    ...         pass
    ...     def execute(self, dataset, target, features):
    ...         # Perform t‑test and return TestResult
    ...         pass
    ...     def calculate_effect_size(self, dataset, target, features):
    ...         # Compute Cohen's d
    ...         pass
    """

    metadata: TestMetadata

    @abc.abstractmethod
    def is_applicable(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> bool:
        """
        Determine whether the test is valid for the given variable types and sample size.

        The method should inspect the dataset's variable types (if available)
        and compare them with the test's `applicable_types`. It should also
        verify that the total sample size meets the test's `min_sample_size`.

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
        bool
            True if the test can be safely applied, False otherwise.

        Raises
        ------
        KeyError
            If `target` or any `feature` is not a column in the dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def check_assumptions(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> List[AssumptionResult]:
        """
        Run assumption checkers (NormalityChecker, VarianceChecker, etc.) for the test.

        The method should perform each assumption check listed in the test's
        `metadata.assumptions` and return a list of `AssumptionResult` objects,
        one for each assumption.

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
        List[AssumptionResult]
            List of assumption check results, in the same order as the
            `metadata.assumptions` list (if possible).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> TestResult:
        """
        Run the statistical test and return a TestResult.

        This is the core method that performs the actual statistical
        computation (e.g., t‑test, ANOVA, chi‑square). It should rely on
        validated data (already checked for applicability and assumptions)
        and produce a fully populated `TestResult` object.

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
        TestResult
            A result object containing the test statistic, p‑value,
            effect size (if computed), and any test‑specific metadata.

        Raises
        ------
        RuntimeError
            If the test cannot be performed due to insufficient data,
            numerical issues, or other runtime problems.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_effect_size(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> Optional[Any]:
        """
        Compute an effect size measure for the test.

        This method may be called after `execute` to provide a standardized
        effect size (e.g., Cohen's d, η², odds ratio). The return type is
        deliberately flexible because effect size representations vary
        widely across tests.

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
            An effect size object (float, dict, dataclass, etc.) or None if
            the test does not define an effect size.
        """
        raise NotImplementedError

    # ------------------ Concrete helper methods ------------------

    def validate_sample_size(
        self,
        dataset: Dataset,
        target: str,
        features: List[str],
    ) -> bool:
        """
        Validate that the total sample size meets the test's minimum requirement.

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
        bool
            True if the dataset's total number of rows (after removing rows
            with missing values in the relevant columns) is at least
            `self.metadata.min_sample_size`.

        Notes
        -----
        This helper is intended to be used inside `is_applicable`. It
        does not consider per‑group sample sizes; tests that require
        per‑group minima must implement additional logic.
        """
        # Extract the columns we need
        columns = [target] + features
        # Drop rows with missing values in any of those columns
        clean_df = dataset.data[columns].dropna()
        return len(clean_df) >= self.metadata.min_sample_size

    def get_groups(
        self,
        dataset: Dataset,
        target: str,
        feature: str,
    ) -> List[pd.Series]:
        """
        Split the target variable into groups based on a categorical feature.

        This is a common operation for tests that compare groups (t‑test,
        ANOVA, etc.). The method extracts the target values for each unique
        level of the feature and returns them as a list of pandas Series.

        Parameters
        ----------
        dataset : Dataset
            Dataset container with the underlying data.
        target : str
            Name of the target (dependent) variable.
        feature : str
            Name of the categorical feature that defines the groups.

        Returns
        -------
        List[pd.Series]
            List of Series, one per unique value of `feature`. Each Series
            contains the `target` values for that group, with missing values
            removed.

        Raises
        ------
        KeyError
            If `target` or `feature` is not a column in the dataset.
        TypeError
            If `feature` is not categorical (optional; currently not enforced).
        """
        df = dataset.data[[target, feature]].dropna()
        groups = []
        for level in df[feature].unique():
            group_data = df.loc[df[feature] == level, target]
            groups.append(group_data.reset_index(drop=True))
        return groups

    def __repr__(self) -> str:
        """Return a concise representation of the test."""
        return f"{self.__class__.__name__}(name='{self.metadata.name}')"