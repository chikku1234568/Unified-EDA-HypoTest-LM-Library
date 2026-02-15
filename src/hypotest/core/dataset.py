"""
Dataset class for holding and validating data for hypothesis testing.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DatasetMetadata:
    """
    Metadata describing a dataset.

    Attributes
    ----------
    n_rows : int
        Number of rows (observations) in the dataset.
    n_columns : int
        Number of columns (variables) in the dataset.
    missing_counts : Dict[str, int]
        Mapping from column name to count of missing (NaN) values.
    dtypes : Dict[str, str]
        Mapping from column name to pandas dtype as string.
    created_at : datetime
        Timestamp when the metadata was computed.
    """
    n_rows: int
    n_columns: int
    missing_counts: Dict[str, int]
    dtypes: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)


class Dataset:
    """
    Container for data with metadata, variable types, and statistics.

    Attributes
    ----------
    data : pd.DataFrame
        The underlying data as a pandas DataFrame.
    metadata : DatasetMetadata
        Computed metadata about the dataset.
    variable_types : Dict[str, Any]
        Mapping from column name to variable type (placeholder).
    statistics : Dict[str, Any]
        Mapping from column name to summary statistics (placeholder).
    """

    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray, List[List[Any]]],
                 columns: Optional[List[str]] = None):
        """
        Initialize a Dataset from various input types.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray, List[List[Any]]]
            Input data. If a DataFrame, used directly. If numpy array or list,
            converted to DataFrame. A 1‑D array or list is treated as a single column.
        columns : Optional[List[str]]
            Column names for array/list data. Ignored if data is a DataFrame.
            If None and data is not a DataFrame, generic names are generated.

        Raises
        ------
        ValueError
            If data cannot be converted to a pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            # Convert to numpy array for consistent shape handling
            arr = np.asarray(data)
            # Ensure 2‑D shape: (n_samples, n_features)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                raise ValueError(
                    f"Input must be 1‑D or 2‑D, got {arr.ndim}‑D array."
                )
            if columns is None:
                n_cols = arr.shape[1]
                columns = [f"col_{i}" for i in range(n_cols)]
            self.data = pd.DataFrame(arr, columns=columns)

        # Ensure we have a DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                f"Could not convert input of type {type(data)} to DataFrame."
            )

        # Compute metadata
        self.metadata = self._compute_metadata()
        # Placeholder for variable types (to be implemented later)
        self.variable_types = {}
        # Placeholder for statistics (to be implemented later)
        self.statistics = {}

    def _compute_metadata(self) -> DatasetMetadata:
        """Compute metadata from the underlying DataFrame."""
        n_rows, n_columns = self.data.shape
        missing_counts = self.data.isna().sum().to_dict()
        dtypes = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        return DatasetMetadata(
            n_rows=n_rows,
            n_columns=n_columns,
            missing_counts=missing_counts,
            dtypes=dtypes
        )

    @property
    def shape(self) -> tuple:
        """
        Return (n_rows, n_columns) as a tuple.

        Returns
        -------
        tuple
            (number of rows, number of columns)
        """
        return self.data.shape

    @property
    def columns(self) -> List[str]:
        """
        Return column names as a list.

        Returns
        -------
        List[str]
            List of column names.
        """
        return list(self.data.columns)

    def __len__(self) -> int:
        """
        Return number of rows in the dataset.

        Returns
        -------
        int
            Number of rows.
        """
        return len(self.data)

    def get_column(self, name: str) -> pd.Series:
        """
        Return a column as a pandas Series.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        pd.Series
            The column data.

        Raises
        ------
        KeyError
            If column does not exist.
        """
        return self.data[name]

    def get_target(self, name: str) -> pd.Series:
        """
        Return a target column as a pandas Series.
        Alias for `get_column` with semantic meaning.

        Parameters
        ----------
        name : str
            Target column name.

        Returns
        -------
        pd.Series
            The target column data.

        Raises
        ------
        KeyError
            If column does not exist.
        """
        return self.get_column(name)

    def get_features(self, names: List[str]) -> pd.DataFrame:
        """
        Return a subset of columns as a DataFrame.

        Parameters
        ----------
        names : List[str]
            List of column names.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the requested columns.

        Raises
        ------
        KeyError
            If any column does not exist.
        """
        return self.data[names]

    def validate(self) -> bool:
        """
        Perform basic validation of the dataset.

        Currently checks:
        - Dataset is not empty
        - All column names are unique
        - No column with all missing values

        Returns
        -------
        bool
            True if dataset passes validation, False otherwise.
        """
        if self.data.empty:
            return False
        if len(self.data.columns) != len(set(self.data.columns)):
            return False
        # Check for columns that are entirely missing
        for col, missing in self.metadata.missing_counts.items():
            if missing == self.metadata.n_rows:
                return False
        return True

    def __repr__(self) -> str:
        """
        Return a concise representation of the Dataset.
        """
        n_rows, n_cols = self.shape
        col_preview = ", ".join(self.columns[:3])
        if len(self.columns) > 3:
            col_preview += ", ..."
        return (
            f"Dataset(shape=({n_rows}, {n_cols}), "
            f"columns=[{col_preview}])"
        )