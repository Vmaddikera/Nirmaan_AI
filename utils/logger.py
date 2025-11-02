#!/usr/bin/env python3
"""
Comprehensive logging system for AI Candidate Analysis
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import traceback
import json

class AnalysisLogger:
    """Centralized logging system for the application"""
    
    def __init__(self, log_level: str = "INFO", log_file: str = "logs/analysis.log"):
        """Initialize the logger with proper configuration"""
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('ai_candidate_analysis')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def log_analysis_start(self, component: str, data_info: Dict[str, Any]):
        """Log the start of an analysis component"""
        self.logger.info(f"Starting {component} analysis")
        self.logger.debug(f"{component} input data: {data_info}")
    
    def log_analysis_success(self, component: str, result_summary: Dict[str, Any]):
        """Log successful completion of analysis"""
        self.logger.info(f"{component} analysis completed successfully")
        self.logger.debug(f"{component} results summary: {result_summary}")
    
    def log_analysis_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Log analysis errors with full context"""
        error_info = {
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.logger.error(f"{component} analysis failed: {error}")
        self.logger.debug(f"Full error details: {json.dumps(error_info, indent=2)}")
        
        return error_info
    
    def log_data_validation(self, validation_result: Dict[str, Any]):
        """Log data validation results"""
        if validation_result.get('valid', False):
            self.logger.info("Data validation passed")
        else:
            self.logger.warning(f"Data validation failed: {validation_result.get('message', 'Unknown error')}")
            self.logger.debug(f"Validation errors: {validation_result.get('errors', [])}")
    
    def log_performance(self, component: str, start_time: datetime, end_time: datetime):
        """Log performance metrics"""
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"{component} completed in {duration:.3f} seconds")
        
        if duration > 5.0:
            self.logger.warning(f"{component} took longer than expected: {duration:.3f} seconds")
    
    def log_system_status(self, status: str, details: Dict[str, Any] = None):
        """Log system status changes"""
        self.logger.info(f"System status: {status}")
        if details:
            self.logger.debug(f"Status details: {details}")
    
    def log_user_action(self, action: str, user_data: Dict[str, Any] = None):
        """Log user actions for audit trail"""
        self.logger.info(f"User action: {action}")
        if user_data:
            # Log only non-sensitive data
            safe_data = {k: v for k, v in user_data.items() if k not in ['transcript_text', 'script_text']}
            self.logger.debug(f"Action data: {safe_data}")

# Global logger instance
logger = AnalysisLogger()

def get_logger() -> AnalysisLogger:
    """Get the global logger instance"""
    return logger

def log_function_call(func):
    """Decorator to automatically log function calls"""
    def wrapper(*args, **kwargs):
        logger.logger.debug(f"Calling {func.__name__} with args: {len(args)}, kwargs: {list(kwargs.keys())}")
        try:
            result = func(*args, **kwargs)
            logger.logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.log_analysis_error(func.__name__, e, {'args_count': len(args), 'kwargs': list(kwargs.keys())})
            raise
    return wrapper
