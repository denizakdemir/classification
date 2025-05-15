"""
Data loading and memory-efficient processing utilities for the classification pipeline.
"""
import pandas as pd
from typing import Optional

class DataLoader:
    """Handles data loading and supports batch/stream processing."""
    def __init__(self, filepath: str, chunksize: Optional[int] = None):
        self.filepath = filepath
        self.chunksize = chunksize

    def load(self):
        if self.chunksize:
            return pd.read_csv(self.filepath, chunksize=self.chunksize)
        else:
            return pd.read_csv(self.filepath)

# Add more data validation and streaming utilities as needed 