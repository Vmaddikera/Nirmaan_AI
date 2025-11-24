"""
Custom exceptions for better error tracking
"""

class AnalysisError(Exception):
    """Base exception for analysis errors"""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class TranscriptError(AnalysisError):
    """Error during transcript processing"""
    pass

class ModelError(AnalysisError):
    """Error loading or using ML models"""
    pass

class ValidationError(AnalysisError):
    """Error validating input data"""
    pass

