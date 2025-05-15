"""
Missing data handling utilities for the classification pipeline.
"""
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class MissingDataHandler:
    """Handles missing data analysis and preprocessing."""
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.missing_stats = None
        self.cols_to_drop = None
        self.missing_indicators = {}

    def analyze_missing(self, df: pd.DataFrame) -> Dict:
        missing_count = df.isnull().sum()
        missing_percentage = missing_count / len(df) * 100
        missing_stats = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percentage.values
        }).sort_values('Missing Percentage', ascending=False)
        self.missing_stats = missing_stats
        self.cols_to_drop = missing_stats[missing_stats['Missing Percentage'] > (self.threshold * 100)]['Column'].tolist()
        return {
            'missing_stats': missing_stats,
            'cols_to_drop': self.cols_to_drop
        }

    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                indicator_name = f"{col}_missing"
                df_processed[indicator_name] = df[col].isnull().astype(int)
                self.missing_indicators[col] = indicator_name
        return df_processed

    def handle_missing(self, df: pd.DataFrame, drop_high_missing: bool = True) -> pd.DataFrame:
        """Process dataframe to handle missing values."""
        # Drop columns with excessive missingness if specified
        if drop_high_missing and self.cols_to_drop:
            df = df.drop(columns=self.cols_to_drop)
            logger.info(f"Dropped {len(self.cols_to_drop)} columns with excessive missing values: {self.cols_to_drop}")
        # Create missing value indicators
        df = self.create_missing_indicators(df)
        return df
    # Add any additional methods as needed from classification_pipeline.py 