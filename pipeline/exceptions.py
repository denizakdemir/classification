"""
Custom exceptions for the classification pipeline.
"""

class DataValidationError(Exception):
    """Raised when input data fails validation checks."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

# Add more custom exceptions as needed 