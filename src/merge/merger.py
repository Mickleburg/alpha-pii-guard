"""High-level span merger for combining regex and NER predictions."""

from typing import List, Tuple, Optional, Dict

from src.merge.span_utils import (
    remove_overlaps_with_priority,
    merge_non_overlapping,
    normalize_spans,
    sort_spans,
)

SpanTuple = Tuple[int, int, str]


class SpanMerger:
    """
    Merge predictions from multiple detectors (regex and NER).
    
    Strategy: regex spans have priority over NER spans.
    Non-conflicting spans are combined.
    Result is always sorted and non-overlapping.
    """
    
    def __init__(self, strategy: str = "regex_priority", validate_text: bool = False):
        """
        Initialize merger.
        
        Args:
            strategy: Merge strategy. Options:
                - "regex_priority": regex spans are kept, NER added if no conflict (default)
                - "ner_priority": NER spans are kept, regex added if no conflict
                - "union": both are merged, overlaps removed (keep longer)
            validate_text: If True, validate spans against text when merging
        """
        if strategy not in ("regex_priority", "ner_priority", "union"):
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.validate_text = validate_text
    
    def merge(
        self,
        regex_spans: List[SpanTuple],
        ner_spans: List[SpanTuple],
        text: Optional[str] = None
    ) -> List[SpanTuple]:
        """
        Merge predictions from regex and NER detectors.
        
        Args:
            regex_spans: Spans from regex detector
            ner_spans: Spans from NER model
            text: Optional original text for validation
            
        Returns:
            Merged list of non-overlapping spans, sorted by position
        """
        # Normalize inputs
        regex_spans = normalize_spans(regex_spans, text=text, drop_invalid=True)
        ner_spans = normalize_spans(ner_spans, text=text, drop_invalid=True)
        
        # Apply merge strategy
        if self.strategy == "regex_priority":
            result = remove_overlaps_with_priority(regex_spans, ner_spans, text=text)
        
        elif self.strategy == "ner_priority":
            result = remove_overlaps_with_priority(ner_spans, regex_spans, text=text)
        
        elif self.strategy == "union":
            result = merge_non_overlapping(regex_spans, ner_spans, text=text)
        
        else:
            # Fallback (should not happen due to __init__ check)
            result = remove_overlaps_with_priority(regex_spans, ner_spans, text=text)
        
        return sort_spans(result)
    
    def __call__(
        self,
        regex_spans: List[SpanTuple],
        ner_spans: List[SpanTuple],
        text: Optional[str] = None
    ) -> List[SpanTuple]:
        """
        Make merger callable.
        
        Args:
            regex_spans: Spans from regex detector
            ner_spans: Spans from NER model
            text: Optional original text for validation
            
        Returns:
            Merged spans
        """
        return self.merge(regex_spans, ner_spans, text=text)


def merge_spans(
    regex_spans: List[SpanTuple],
    ner_spans: List[SpanTuple],
    strategy: str = "regex_priority",
    text: Optional[str] = None
) -> List[SpanTuple]:
    """
    Convenience function to merge spans without creating SpanMerger instance.
    
    Args:
        regex_spans: Spans from regex detector
        ner_spans: Spans from NER model
        strategy: Merge strategy ("regex_priority", "ner_priority", "union")
        text: Optional original text for validation
        
    Returns:
        Merged list of non-overlapping spans, sorted by position
    """
    merger = SpanMerger(strategy=strategy)
    return merger.merge(regex_spans, ner_spans, text=text)


# Backward compatibility: alias for existing code that might use MergeResolver directly
def resolve_conflicts(
    regex_spans: List[SpanTuple],
    ner_spans: List[SpanTuple],
    text: Optional[str] = None
) -> List[SpanTuple]:
    """
    Backward-compatible wrapper for existing code.
    
    This function maintains compatibility with resolver.py API while using
    the unified merge logic.
    
    Args:
        regex_spans: Spans from regex detector
        ner_spans: Spans from NER model
        text: Optional original text for validation
        
    Returns:
        Merged spans (regex priority)
    """
    return merge_spans(regex_spans, ner_spans, strategy="regex_priority", text=text)
