"""
Logging utilities for fire detection system
"""
import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: Optional[str] = None
):
    """
    Setup logger configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: Log rotation size
        retention: Log retention period
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    return logger

def get_logger(name: str = None):
    """Get logger instance"""
    if name:
        return logger.bind(name=name)
    return logger

# Setup default logger
default_log_file = "outputs/logs/fire_detection.log"
setup_logger(log_file=default_log_file)

