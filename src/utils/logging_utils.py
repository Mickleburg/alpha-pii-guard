"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global loggers cache
_loggers = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                If None, uses LOG_LEVEL from environment
    
    Returns:
        Configured Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Only add handlers if logger doesn't have them yet
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Formatter
        log_format = os.getenv("LOG_FORMAT", "text").lower()
        if log_format == "json":
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    _loggers[name] = logger
    return logger


def setup_logging(
    config_dict: Optional[dict] = None,
    log_dir: Optional[str | Path] = None,
    level: str = "INFO"
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        config_dict: Configuration dictionary with logging settings
        log_dir: Directory to save log files
        level: Default log level
    """
    # Set root logger level
    logging.getLogger().setLevel(level)
    
    # Create log directory if needed
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = log_path / "pii_ner.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        
        log_format = os.getenv("LOG_FORMAT", "text").lower()
        if log_format == "json":
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Configure specific loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def log_config(config: dict, logger: logging.Logger) -> None:
    """
    Log configuration dictionary in a readable format.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration loaded:")
    
    def log_dict(d: dict, prefix: str = "  ") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)
