"""
Statistical description module for hypothesis testing.

Provides Statistics dataclass and StatisticsCalculator for computing
descriptive statistics for continuous, categorical, binary, and ordinal variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union
from .type_detector import VariableType


@dataclass
class Statistics:
    """
    Comprehensive descriptive statistics for a single variable.

    Attributes
    ----------
    mean : Optional[float]
        Arithmetic mean. Computed for continuous variables.
    median : Optional[float]
        Median (50th percentile). Computed for continuous variables.
    std : Optional[float]
        Standard deviation (sample). Computed for continuous variables.
    min : Optional[float]
        Minimum value. Computed for continuous variables.
    max : Optional[float]
        Maximum value. Computed for continuous variables.
    quartiles : Optional[List[float]]
        Three quartiles [Q1 (25%), Q2 (50%), Q3 (75%)]. Computed for continuous.
    mode : Optional[Any]
        Most frequent value(s). For categorical/binary/ordinal variables.
    unique_count : int
        Number of distinct non‑missing values.
    missing_count : int
        Number of missing (NaN/None) values.
    missing_percentage : float
        Percentage of missing values relative to total observations.

    Notes
    -----
    Fields that are not applicable for a given variable type are set to None.
    """

    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    quartiles: Optional[List[float]] = None
    mode: Optional[Any] = None
    unique_count: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0


class StatisticsCalculator:
    """
    Compute descriptive statistics for pandas Series based on variable type.

    Uses the VariableType classification from `type_detector` to decide
    which statistics are appropriate (continuous vs categorical/binary/ordinal).

    Missing values are handled safely: they are excluded from computations
    of mean, median, etc., but counted in missing_count and missing_percentage.

    Methods
    -------
    calculate(series: pd.Series, var_type: VariableType) -> Statistics
        Compute statistics for a single series.

    calculate_all(df: pd.DataFrame, variable_types: Dict[str, VariableType])
        -> Dict[str, Statistics]
        Compute statistics for all columns of a DataFrame.

    Private helpers
    ---------------
    _calculate_continuous_stats(series: pd.Series) -> Dict[str, Any]
    _calculate_categorical_stats(series: pd.Series) -> Dict[str, Any]
    _calculate_missing_stats(series: pd.Series) -> Dict[str, Any]
    _calculate_unique_count(series: pd.Series) -> int
    """

    def __init__(self):
        """Initialize the calculator with default settings."""
        pass

    def calculate(self, series: pd.Series, var_type: VariableType) -> Statistics:
        """
        Compute descriptive statistics for a pandas Series.

        Parameters
        ----------
        series : pd.Series
            The data column.
        var_type : VariableType
            Type of the variable (continuous, categorical, binary, ordinal).

        Returns
        -------
        Statistics
            Dataclass containing all computed statistics.

        Raises
        ------
        ValueError
            If series is empty or var_type is not a valid VariableType.
        """
        if series.empty:
            raise ValueError("Cannot compute statistics on an empty series")

        # Compute missing statistics (always)
        missing_stats = self._calculate_missing_stats(series)

        # Compute type‑specific statistics
        if var_type == VariableType.CONTINUOUS:
            cont_stats = self._calculate_continuous_stats(series)
            cat_stats = {}
        else:
            # categorical, binary, ordinal
            cat_stats = self._calculate_categorical_stats(series)
            cont_stats = {}

        # Compute unique count (applicable to all types)
        unique_count = self._calculate_unique_count(series)

        # Merge into Statistics object
        stats = Statistics(
            mean=cont_stats.get("mean"),
            median=cont_stats.get("median"),
            std=cont_stats.get("std"),
            min=cont_stats.get("min"),
            max=cont_stats.get("max"),
            quartiles=cont_stats.get("quartiles"),
            mode=cat_stats.get("mode"),
            unique_count=unique_count,
            missing_count=missing_stats["missing_count"],
            missing_percentage=missing_stats["missing_percentage"],
        )
        return stats

    def calculate_all(
        self, df: pd.DataFrame, variable_types: Dict[str, VariableType]
    ) -> Dict[str, Statistics]:
        """
        Compute statistics for each column of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        variable_types : Dict[str, VariableType]
            Mapping from column name to its VariableType.
            Must contain a key for every column of df.

        Returns
        -------
        Dict[str, Statistics]
            Dictionary mapping column names to their Statistics.

        Raises
        ------
        ValueError
            If a column in df is not present in variable_types.
        """
        result = {}
        for col in df.columns:
            if col not in variable_types:
                raise ValueError(
                    f"Variable type not provided for column '{col}'"
                )
            result[col] = self.calculate(df[col], variable_types[col])
        return result

    def _calculate_continuous_stats(self, series: pd.Series) -> Dict[str, Any]:
        """
        Compute continuous‑variable statistics.

        Parameters
        ----------
        series : pd.Series
            Numeric data (may contain missing values).

        Returns
        -------
        dict
            With keys: mean, median, std, min, max, quartiles.
            All values are floats except quartiles (list of three floats).
            If the series contains no non‑missing numeric values, all fields
            are None.
        """
        # Drop missing values for numeric computations
        clean = series.dropna()
        if clean.empty:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "quartiles": None,
            }

        # Ensure numeric dtype
        if not pd.api.types.is_numeric_dtype(clean.dtype):
            clean = pd.to_numeric(clean, errors="coerce")
            clean = clean.dropna()
            if clean.empty:
                return {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "quartiles": None,
                }

        values = clean.values
        stats = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values, ddof=1)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
        # quartiles: 25%, 50%, 75%
        quartiles = np.nanpercentile(values, [25, 50, 75])
        stats["quartiles"] = [float(q) for q in quartiles]
        return stats

    def _calculate_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """
        Compute categorical‑variable statistics.

        Parameters
        ----------
        series : pd.Series
            Data of any dtype (including numeric but treated as categorical).

        Returns
        -------
        dict
            With keys: mode, unique_count.
            mode is the most frequent value (if tie, the first encountered).
            If the series contains no non‑missing values, mode is None and
            unique_count is 0.
        """
        # Drop missing values for categorical analysis
        clean = series.dropna()
        if clean.empty:
            return {"mode": None, "unique_count": 0}

        unique_count = clean.nunique()
        # Compute mode(s)
        value_counts = clean.value_counts()
        if value_counts.empty:
            mode_val = None
        else:
            # If multiple modes, pick the first (pandas .mode() returns Series)
            mode_val = value_counts.index[0]
            # If there is a tie, value_counts sorted descending, first is one of them

        return {"mode": mode_val, "unique_count": int(unique_count)}

    def _calculate_missing_stats(self, series: pd.Series) -> Dict[str, Any]:
        """
        Compute missing‑value statistics.

        Parameters
        ----------
        series : pd.Series
            Any data.

        Returns
        -------
        dict
            With keys: missing_count, missing_percentage.
        """
        total = len(series)
        missing_count = int(series.isna().sum())
        missing_percentage = (missing_count / total * 100.0) if total > 0 else 0.0
        return {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }

    def _calculate_unique_count(self, series: pd.Series) -> int:
        """
        Count distinct non‑missing values in a series.

        Parameters
        ----------
        series : pd.Series
            Any data.

        Returns
        -------
        int
            Number of unique non‑missing values.
        """
        clean = series.dropna()
        if clean.empty:
            return 0
        return int(clean.nunique())


# Legacy functions (kept for backward compatibility, but deprecated)
def compute_statistics(
    data: Union[pd.Series, np.ndarray], axis: Optional[int] = None
) -> Dict[str, float]:
    """
    Legacy function for basic descriptive statistics.

    Deprecated. Use StatisticsCalculator instead.

    Parameters
    ----------
    data : Union[pd.Series, np.ndarray]
        Input data.
    axis : Optional[int]
        Not implemented.

    Returns
    -------
    dict
        Dictionary with keys: mean, std, median, min, max, n.
    """
    import warnings
    warnings.warn(
        "compute_statistics is deprecated; use StatisticsCalculator",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(data, pd.Series):
        d = data.values
    else:
        d = np.asarray(data)

    if axis is not None:
        raise NotImplementedError("Axis parameter not yet implemented.")

    stats = {
        "mean": np.nanmean(d),
        "std": np.nanstd(d, ddof=1),
        "median": np.nanmedian(d),
        "min": np.nanmin(d),
        "max": np.nanmax(d),
        "n": np.sum(~np.isnan(d)),
    }
    return stats


def correlation(
    x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]
) -> tuple[float, float]:
    """
    Compute Pearson correlation and p‑value.

    Parameters
    ----------
    x, y : array‑like
        Two variables of equal length.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    p : float
        Two‑tailed p‑value.
    """
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return r, p