"""Regex-based PII detector."""

import re
import time
from typing import List, Tuple, Optional, Dict, Set

from src.utils.logging_utils import get_logger
from src.regex.patterns import get_pattern_registry, RegexPatterns
from src.regex.context_rules import get_rule_registry

logger = get_logger(__name__)

# Type alias
SpanTuple = Tuple[int, int, str]


class RegexPIIDetector:
    """Regex-based detector for PII in text."""
    
    def __init__(self, use_context_rules: bool = True, timeout_ms: float = 1000):
        """
        Initialize detector.
        
        Args:
            use_context_rules: Whether to apply context filtering rules
            timeout_ms: Timeout in milliseconds for pattern matching
        """
        self.pattern_registry = get_pattern_registry()
        self.rule_registry = get_rule_registry() if use_context_rules else None
        self.timeout_ms = timeout_ms
        self.use_context_rules = use_context_rules
    
    def predict(self, text: str) -> List[SpanTuple]:
        """
        Detect PII in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, category) tuples, sorted by start, no overlaps
        """
        if not text:
            return []
        
        # Find all candidates
        candidates = self._find_candidates(text)
        
        # Filter by context rules
        if self.use_context_rules and self.rule_registry:
            candidates = self._filter_by_context(text, candidates)
        
        # Remove overlaps (keep longer spans, prefer earlier in text)
        candidates = self._remove_overlaps(candidates)
        
        # Sort by start position
        candidates.sort(key=lambda x: x[0])
        
        # Final validation
        candidates = self._validate_spans(text, candidates)
        
        return candidates
    
    def batch_predict(self, texts: List[str]) -> List[List[SpanTuple]]:
        """
        Detect PII in multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of span lists (parallel to input)
        """
        return [self.predict(text) for text in texts]
    
    def _find_candidates(self, text: str) -> List[SpanTuple]:
        """
        Find all potential PII matches using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, category) tuples (may include overlaps)
        """
        candidates = []
        start_time = time.time() * 1000  # Convert to ms
        
        for category, patterns in self.pattern_registry.items():
            # Check timeout
            elapsed = (time.time() * 1000) - start_time
            if elapsed > self.timeout_ms:
                logger.warning(f"Timeout reached for category {category}")
                break
            
            for pattern in patterns:
                try:
                    for match in pattern.finditer(text):
                        span = (match.start(), match.end(), category)
                        candidates.append(span)
                
                except Exception as e:
                    logger.warning(f"Error matching pattern in {category}: {e}")
                    continue
        
        return candidates
    
    def _filter_by_context(
        self,
        text: str,
        candidates: List[SpanTuple]
    ) -> List[SpanTuple]:
        """
        Filter candidates by context rules.
        
        Args:
            text: Original text
            candidates: Candidate spans
            
        Returns:
            Filtered spans
        """
        filtered = []
        
        for start, end, category in candidates:
            if self.rule_registry.should_accept(category, text, start, end):
                filtered.append((start, end, category))
        
        return filtered
    
    def _remove_overlaps(self, spans: List[SpanTuple]) -> List[SpanTuple]:
        """
        Remove overlapping spans.
        
        Strategy:
        - Sort by (start, -length) to process longer spans first
        - Keep spans that don't overlap with already-kept spans
        
        Args:
            spans: Potentially overlapping spans
            
        Returns:
            Non-overlapping spans
        """
        if not spans:
            return []
        
        # Sort by start, then by length descending (keep longer spans)
        sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
        
        result = []
        used_positions = set()
        
        for start, end, category in sorted_spans:
            # Check if any position in this span is already used
            overlaps = any(pos in used_positions for pos in range(start, end))
            
            if not overlaps:
                result.append((start, end, category))
                for pos in range(start, end):
                    used_positions.add(pos)
        
        return result
    
    def _validate_spans(
        self,
        text: str,
        spans: List[SpanTuple]
    ) -> List[SpanTuple]:
        """
        Validate that spans are valid.
        
        Checks:
        - 0 <= start < end <= len(text)
        - Span text is not all whitespace
        
        Args:
            text: Original text
            spans: Spans to validate
            
        Returns:
            Valid spans
        """
        valid = []
        text_len = len(text)
        
        for start, end, category in spans:
            # Bounds check
            if not (0 <= start < end <= text_len):
                logger.warning(f"Invalid span bounds: ({start}, {end}) for text length {text_len}")
                continue
            
            # Check text is not all whitespace
            span_text = text[start:end]
            if not span_text or not span_text.strip():
                logger.warning(f"Span contains only whitespace: [{start}:{end}]")
                continue
            
            valid.append((start, end, category))
        
        return valid
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported PII categories."""
        return RegexPatterns.get_categories()
    
    def get_patterns_for_category(self, category: str) -> int:
        """Get number of patterns for a category."""
        patterns = self.pattern_registry.get(category, [])
        return len(patterns)


def create_detector(
    use_context_rules: bool = True,
    timeout_ms: float = 1000
) -> RegexPIIDetector:
    """
    Factory function to create a detector instance.
    
    Args:
        use_context_rules: Whether to apply context filtering
        timeout_ms: Timeout in milliseconds
        
    Returns:
        RegexPIIDetector instance
    """
    return RegexPIIDetector(
        use_context_rules=use_context_rules,
        timeout_ms=timeout_ms
    )
