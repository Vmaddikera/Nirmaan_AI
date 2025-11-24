"""
Simple logger for tracking application events and errors
"""
import logging
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('RubricAnalyzer')

def log_info(message):
    """Log informational message"""
    logger.info(message)

def log_error(message, exception=None):
    """Log error message with optional exception"""
    if exception:
        logger.error(f"{message} | Error: {str(exception)}")
    else:
        logger.error(message)

def log_analysis(transcript_length, duration, score):
    """Log analysis results"""
    logger.info(f"Analysis Complete | Length: {transcript_length} chars | Duration: {duration}s | Score: {score}/100")

