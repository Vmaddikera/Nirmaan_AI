#!/usr/bin/env python3
"""
Production-ready error handling for AI Candidate Analysis
"""

import traceback
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .exceptions import *

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger('errors')
        self.analysis_logger = logging.getLogger('analysis')
    
    def handle_analysis_error(self, error: Exception, component: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle analysis-specific errors with detailed logging"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # Log based on error type
        if isinstance(error, AnalysisException):
            self.logger.error(f"Analysis error in {component}: {error}")
            self.logger.debug(f"Full error details: {error_info}")
        else:
            self.logger.error(f"Unexpected error in {component}: {error}")
            self.logger.debug(f"Full error details: {error_info}")
        
        # Return sanitized error info for API response
        return {
            'error': str(error),
            'component': component,
            'error_code': getattr(error, 'error_code', 'UNKNOWN_ERROR'),
            'timestamp': error_info['timestamp']
        }
    
    def handle_system_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle system-level errors"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.critical(f"System error in {operation}: {error}")
        self.logger.debug(f"Full error details: {error_info}")
        
        return {
            'error': 'Internal system error',
            'operation': operation,
            'error_code': 'SYSTEM_ERROR',
            'timestamp': error_info['timestamp']
        }
    
    def handle_validation_error(self, error: Exception, validation_type: str, data_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle data validation errors"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': validation_type,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'data_info': data_info or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.warning(f"Validation error in {validation_type}: {error}")
        self.logger.debug(f"Full error details: {error_info}")
        
        return {
            'error': f'Data validation failed: {str(error)}',
            'validation_type': validation_type,
            'error_code': 'VALIDATION_ERROR',
            'timestamp': error_info['timestamp']
        }
    
    def handle_model_error(self, error: Exception, model_name: str, operation: str = None) -> Dict[str, Any]:
        """Handle AI model-related errors"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'operation': operation or 'unknown',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Model error in {model_name}: {error}")
        self.logger.debug(f"Full error details: {error_info}")
        
        return {
            'error': f'Model {model_name} failed: {str(error)}',
            'model_name': model_name,
            'operation': operation,
            'error_code': 'MODEL_ERROR',
            'timestamp': error_info['timestamp']
        }
    
    def create_error_response(self, error_dict: Dict[str, Any], status_code: int = 500) -> tuple:
        """Create standardized error response for API"""
        
        response = {
            'success': False,
            'error': error_dict['error'],
            'error_code': error_dict.get('error_code', 'UNKNOWN_ERROR'),
            'timestamp': error_dict['timestamp']
        }
        
        # Add component info if available
        if 'component' in error_dict:
            response['component'] = error_dict['component']
        
        return response, status_code

# Global error handler instance
error_handler = ErrorHandler()

def handle_exception(func):
    """Decorator to handle exceptions in functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnalysisException as e:
            return error_handler.handle_analysis_error(e, e.component, e.details)
        except Exception as e:
            return error_handler.handle_system_error(e, func.__name__)
    return wrapper
