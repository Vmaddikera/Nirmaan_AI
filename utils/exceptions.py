#!/usr/bin/env python3
"""
Custom exception classes for AI Candidate Analysis
"""

class AnalysisException(Exception):
    """Base exception for all analysis-related errors"""
    
    def __init__(self, message: str, component: str = None, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.component = component
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None
    
    def to_dict(self):
        """Convert exception to dictionary for logging"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'component': self.component,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }

class DataValidationError(AnalysisException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, validation_errors: list = None):
        super().__init__(message, component="DataValidator", error_code="VALIDATION_FAILED")
        self.validation_errors = validation_errors or []

class EmotionAnalysisError(AnalysisException):
    """Raised when emotion analysis fails"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, component="EmotionAnalyzer", error_code="EMOTION_ANALYSIS_FAILED", details=details)

class GazeAnalysisError(AnalysisException):
    """Raised when gaze analysis fails"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, component="GazeAnalyzer", error_code="GAZE_ANALYSIS_FAILED", details=details)

class NLPAnalysisError(AnalysisException):
    """Raised when NLP analysis fails"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, component="NLPAnalyzer", error_code="NLP_ANALYSIS_FAILED", details=details)

class TranscriptAnalysisError(AnalysisException):
    """Raised when transcript analysis fails"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, component="TranscriptAnalyzer", error_code="TRANSCRIPT_ANALYSIS_FAILED", details=details)

class AssessmentError(AnalysisException):
    """Raised when final assessment fails"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, component="CandidateAssessor", error_code="ASSESSMENT_FAILED", details=details)

class ModelLoadingError(AnalysisException):
    """Raised when AI models fail to load"""
    
    def __init__(self, message: str, model_name: str = None, details: dict = None):
        super().__init__(message, component="ModelLoader", error_code="MODEL_LOADING_FAILED", details=details)
        self.model_name = model_name

class DataProcessingError(AnalysisException):
    """Raised when data processing fails"""
    
    def __init__(self, message: str, processing_step: str = None, details: dict = None):
        super().__init__(message, component="DataProcessor", error_code="DATA_PROCESSING_FAILED", details=details)
        self.processing_step = processing_step

class JSONSerializationError(AnalysisException):
    """Raised when JSON serialization fails"""
    
    def __init__(self, message: str, data_type: str = None, details: dict = None):
        super().__init__(message, component="JSONSerializer", error_code="JSON_SERIALIZATION_FAILED", details=details)
        self.data_type = data_type

def handle_analysis_exception(func):
    """Decorator to handle analysis exceptions with proper logging"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnalysisException as e:
            # Re-raise analysis exceptions as-is
            raise
        except Exception as e:
            # Convert generic exceptions to analysis exceptions
            component = getattr(func, '__self__', {}).get('__class__', {}).get('__name__', 'Unknown')
            raise AnalysisException(
                f"Unexpected error in {component}: {str(e)}",
                component=component,
                details={'original_error': str(e), 'function': func.__name__}
            )
    return wrapper
