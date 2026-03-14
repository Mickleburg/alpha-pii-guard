"""
PII NER Detection System

Production-grade Named Entity Recognition for Personally Identifiable Information
in Russian text using a hybrid approach combining regex patterns and BERT-based NER.
"""

__version__ = "1.0.0"
__author__ = "NER Team"

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid circular dependencies
__all__ = [
    "logger",
    "__version__",
]
