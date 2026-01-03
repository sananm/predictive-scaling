"""
Base classes for feature extraction.

Provides:
- Abstract base extractor class
- Common utilities for all extractors
"""

from abc import ABC, abstractmethod

import pandas as pd

from .config import FeatureConfig


class BaseExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All extractors must implement the extract() method
    which takes a DataFrame and returns a DataFrame of features.
    """

    def __init__(self, config: FeatureConfig) -> None:
        """
        Initialize the extractor.

        Args:
            config: Feature configuration
        """
        self.config = config
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        return self._feature_names

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from input DataFrame.

        Args:
            df: Input DataFrame with timestamp index and metric columns

        Returns:
            DataFrame with extracted features
        """
        pass

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If input is invalid
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

    def _add_prefix(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add prefix to all column names."""
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df
