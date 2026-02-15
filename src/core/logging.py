"""
Structured logging setup for the application.
Provides consistent log formatting across all modules.
"""
import logging
import sys
from typing import Optional

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure application logging with structured output.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    log_level = level or settings.log_level
    
    # Default format includes timestamp, level, module, and message
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger("founder-context-engine")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


# Global logger instance
logger = setup_logging()
