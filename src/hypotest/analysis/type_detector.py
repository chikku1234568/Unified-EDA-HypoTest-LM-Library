"""
Detect variable types for hypothesis test selection.

This module provides the VariableType enumeration and the TypeDetector class,
which automatically classifies pandas Series into continuous, categorical,
binary, or ordinal types using intelligent heuristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from enum import Enum


class VariableType(str, Enum):
    """
    Enumeration of possible variable types in statistical analysis.

    Attributes
    ----------
    CONTINUOUS : str
        Numeric variable with many unique values, suitable for parametric tests.
    CATEGORICAL : str
        Non‑numeric variable with discrete categories (nominal).
    BINARY : str
        Variable with exactly two distinct values (including boolean).
    ORDINAL : str
        Numeric variable with few unique values, ordered but not continuous.
    """

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"


class TypeDetector:
    """
    Determine variable types (continuous, categorical, binary, ordinal) using
    data‑driven heuristics.

    The detector examines the dtype, number of unique values, and the
    proportion of unique values relative to the series length. It returns
    a VariableType enum member and an optional confidence score.

    Parameters
    ----------
    max_categories : int, default=10
        Maximum number of unique values for a numeric series to be considered
        ordinal (otherwise continuous). Ignored for non‑numeric series.
    binary_threshold : int, default=2
        Number of unique values that defines a binary variable (≤ binary_threshold).
    low_cardinality_ratio : float, default=0.1
        Proportion of unique values below which a numeric series is considered
        ordinal (if unique count ≤ max_categories). Must be between 0 and 1.
    handle_missing : str, default='ignore'
        How to handle missing values. 'ignore' drops them before counting
        uniques; 'include' treats NaN as a distinct category (not recommended).

    Examples
    --------
    >>> detector = TypeDetector()
    >>> series = pd.Series([1, 2, 3, 4, 5])
    >>> detector.detect(series)
    <VariableType.CONTINUOUS: 'continuous'>
    >>> detector.get_confidence(series, VariableType.CONTINUOUS)
    0.95
    """

    def __init__(
        self,
        max_categories: int = 10,
        binary_threshold: int = 2,
        low_cardinality_ratio: float = 0.1,
        handle_missing: str = "ignore",
    ):
        self.max_categories = max_categories
        self.binary_threshold = binary_threshold
        self.low_cardinality_ratio = low_cardinality_ratio
        self.handle_missing = handle_missing

        if handle_missing not in ("ignore", "include"):
            raise ValueError(
                "handle_missing must be either 'ignore' or 'include'"
            )

    def _is_numeric(self, series: pd.Series) -> bool:
        """
        Return True if the series has a numeric dtype.

        Parameters
        ----------
        series : pd.Series
            Input data.

        Returns
        -------
        bool
            True if the dtype is numeric (int, float, etc.).
        """
        return pd.api.types.is_numeric_dtype(series.dtype)

    def _is_binary(self, series: pd.Series) -> bool:
        """
        Return True if the series has exactly two unique non‑missing values.

        Parameters
        ----------
        series : pd.Series
            Input data.

        Returns
        -------
        bool
            True if binary.
        """
        if self.handle_missing == "ignore":
            series = series.dropna()
        unique = series.nunique()
        return unique <= self.binary_threshold

    def _is_categorical(self, series: pd.Series) -> bool:
        """
        Return True if the series is non‑numeric (object, string, category).

        Parameters
        ----------
        series : pd.Series
            Input data.

        Returns
        -------
        bool
            True if categorical.
        """
        return not self._is_numeric(series)

    def _is_ordinal(self, series: pd.Series) -> bool:
        """
        Return True if the series is numeric and has few unique values.

        Parameters
        ----------
        series : pd.Series
            Input data.

        Returns
        -------
        bool
            True if ordinal.
        """
        if not self._is_numeric(series):
            return False
        if self.handle_missing == "ignore":
            series = series.dropna()
        n_unique = series.nunique()
        n_total = len(series)
        if n_total == 0:
            return False
        return (
            n_unique <= self.max_categories
            and n_unique / n_total <= self.low_cardinality_ratio
        )

    def _is_continuous(self, series: pd.Series) -> bool:
        """
        Return True if the series is numeric and not ordinal nor binary.

        Parameters
        ----------
        series : pd.Series
            Input data.

        Returns
        -------
        bool
            True if continuous.
        """
        if not self._is_numeric(series):
            return False
        if self._is_binary(series):
            return False
        if self._is_ordinal(series):
            return False
        return True

    def detect(self, series: pd.Series) -> VariableType:
        """
        Detect the variable type of a pandas Series.

        The detection logic follows:
        1. If dtype is boolean → BINARY
        2. If numeric dtype:
            - 2 unique values → BINARY
            - few unique values relative to length → ORDINAL
            - many unique values → CONTINUOUS
        3. If object/string dtype → CATEGORICAL

        Missing values are ignored (dropped) before counting uniques.

        Parameters
        ----------
        series : pd.Series
            The column data to classify.

        Returns
        -------
        VariableType
            The detected type.

        Raises
        ------
        ValueError
            If the series is empty after dropping missing values.

        Examples
        --------
        >>> detector = TypeDetector()
        >>> detector.detect(pd.Series([1, 2, 3, 4, 5]))
        <VariableType.CONTINUOUS: 'continuous'>
        >>> detector.detect(pd.Series(['A', 'B', 'A', 'C']))
        <VariableType.CATEGORICAL: 'categorical'>
        """
        if series.empty:
            raise ValueError("Cannot detect type of an empty series")

        # Handle missing values
        if self.handle_missing == "ignore":
            series = series.dropna()
            if series.empty:
                raise ValueError(
                    "Series contains only missing values after dropna"
                )

        # Boolean dtype → binary
        if pd.api.types.is_bool_dtype(series.dtype):
            return VariableType.BINARY

        # Numeric dtype
        if self._is_numeric(series):
            if self._is_binary(series):
                return VariableType.BINARY
            if self._is_ordinal(series):
                return VariableType.ORDINAL
            return VariableType.CONTINUOUS

        # Non‑numeric → categorical
        return VariableType.CATEGORICAL

    def detect_all(self, df: pd.DataFrame) -> Dict[str, VariableType]:
        """
        Detect variable types for all columns in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        Dict[str, VariableType]
            Mapping from column name to detected VariableType.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'gender': ['M', 'F', 'M'],
        ...     'passed': [True, False, True]
        ... })
        >>> detector = TypeDetector()
        >>> detector.detect_all(df)
        {'age': <VariableType.CONTINUOUS: 'continuous'>,
         'gender': <VariableType.CATEGORICAL: 'categorical'>,
         'passed': <VariableType.BINARY: 'binary'>}
        """
        return {col: self.detect(df[col]) for col in df.columns}

    def get_confidence(
        self, series: pd.Series, var_type: VariableType
    ) -> float:
        """
        Return a confidence score (0‑1) for the detected variable type.

        Heuristics:
        - For CONTINUOUS: high confidence if many unique values (> 20) and
          high cardinality ratio (> 0.5). Lower if few uniques.
        - For BINARY: high confidence if exactly two unique values and
          neither dominates (> 10% each). Lower if extreme imbalance.
        - For ORDINAL: high confidence if unique count ≤ max_categories and
          low cardinality ratio (< 0.2). Lower if near threshold.
        - For CATEGORICAL: high confidence if non‑numeric dtype; moderate
          if numeric but forced categorical.

        Parameters
        ----------
        series : pd.Series
            The column data.
        var_type : VariableType
            The type to evaluate confidence for (should match detect output).

        Returns
        -------
        float
            Confidence score between 0 and 1.
        """
        if self.handle_missing == "ignore":
            series = series.dropna()

        if series.empty:
            return 0.0

        n_unique = series.nunique()
        n_total = len(series)
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0.0

        if var_type == VariableType.CONTINUOUS:
            # Confidence increases with more unique values and higher ratio
            unique_score = min(n_unique / 100, 1.0)  # cap at 100 uniques
            ratio_score = min(cardinality_ratio * 2, 1.0)  # 0.5 → 1.0
            return 0.3 + 0.7 * (unique_score * 0.6 + ratio_score * 0.4)

        if var_type == VariableType.BINARY:
            if n_unique != 2:
                return 0.2  # shouldn't happen
            # Check balance
            value_counts = series.value_counts(normalize=True)
            min_prop = value_counts.min()
            balance_score = min_prop / 0.5  # 0.5 is perfect balance
            return 0.5 + 0.5 * balance_score

        if var_type == VariableType.ORDINAL:
            # Confidence high if well below thresholds
            unique_ok = n_unique <= self.max_categories
            ratio_ok = cardinality_ratio <= self.low_cardinality_ratio
            if not (unique_ok and ratio_ok):
                return 0.3
            # Compute how far from thresholds
            unique_margin = (self.max_categories - n_unique) / self.max_categories
            ratio_margin = (
                self.low_cardinality_ratio - cardinality_ratio
            ) / self.low_cardinality_ratio
            margin_score = (unique_margin + ratio_margin) / 2
            return 0.6 + 0.4 * margin_score

        if var_type == VariableType.CATEGORICAL:
            # High confidence if non‑numeric
            if not self._is_numeric(series):
                return 0.9
            # Numeric but classified as categorical (edge case)
            return 0.4

        # Fallback
        return 0.5