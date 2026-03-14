"""Merge resolution logic for combining regex and NER predictions."""

from typing import List, Tuple, Set, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


class ConflictResolver:
    """Resolve conflicts between regex and NER detections."""
    
    @staticmethod
    def spans_overlap(span1: SpanTuple, span2: SpanTuple) -> bool:
        """
        Check if two spans overlap.
        
        Args:
            span1: First span (start, end, category)
            span2: Second span (start, end, category)
            
        Returns:
            True if spans overlap
        """
        start1, end1, _ = span1
        start2, end2, _ = span2
        
        # Overlap if one starts before the other ends
        return not (end1 <= start2 or end2 <= start1)
    
    @staticmethod
    def span_contains(outer: SpanTuple, inner: SpanTuple) -> bool:
        """
        Check if outer span fully contains inner span.
        
        Args:
            outer: Potentially containing span
            inner: Potentially contained span
            
        Returns:
            True if outer contains inner
        """
        start_outer, end_outer, _ = outer
        start_inner, end_inner, _ = inner
        
        return start_outer <= start_inner and end_inner <= end_outer
    
    @staticmethod
    def spans_identical(span1: SpanTuple, span2: SpanTuple) -> bool:
        """
        Check if spans are identical.
        
        Args:
            span1: First span
            span2: Second span
            
        Returns:
            True if identical (same start, end, category)
        """
        start1, end1, cat1 = span1
        start2, end2, cat2 = span2
        
        return start1 == start2 and end1 == end2 and cat1 == cat2
    
    @classmethod
    def resolve_conflicts(
        cls,
        regex_spans: List[SpanTuple],
        ner_spans: List[SpanTuple]
    ) -> List[SpanTuple]:
        """
        Resolve conflicts between regex and NER spans.
        
        Strategy:
        1. Regex spans have priority - keep all regex spans
        2. For each NER span, keep it only if it doesn't conflict with any regex span
        3. Remove duplicates
        4. Sort by start position
        
        Args:
            regex_spans: Spans from regex detector
            ner_spans: Spans from NER model
            
        Returns:
            Merged spans with conflicts resolved
        """
        # Start with regex spans (highest priority)
        result = list(regex_spans)
        result_positions = cls._get_covered_positions(result)
        
        # Add NER spans that don't conflict with regex
        for ner_span in ner_spans:
            ner_start, ner_end, _ = ner_span
            
            # Check if NER span overlaps with any regex span
            conflicts = False
            for regex_span in regex_spans:
                if cls.spans_overlap(regex_span, ner_span):
                    conflicts = True
                    logger.debug(
                        f"NER span ({ner_start}, {ner_end}) conflicts with "
                        f"regex span {regex_span} - skipping NER"
                    )
                    break
            
            if not conflicts:
                result.append(ner_span)
        
        # Remove duplicates
        result = cls._deduplicate_spans(result)
        
        # Sort by start position
        result.sort(key=lambda x: x[0])
        
        return result
    
    @classmethod
    def _get_covered_positions(cls, spans: List[SpanTuple]) -> Set[int]:
        """Get all character positions covered by spans."""
        positions = set()
        for start, end, _ in spans:
            for pos in range(start, end):
                positions.add(pos)
        return positions
    
    @classmethod
    def _deduplicate_spans(cls, spans: List[SpanTuple]) -> List[SpanTuple]:
        """Remove duplicate spans."""
        seen = set()
        result = []
        
        for start, end, category in spans:
            span_key = (start, end, category)
            if span_key not in seen:
                seen.add(span_key)
                result.append((start, end, category))
        
        return result
    
    @classmethod
    def remove_nested_overlaps(
        cls,
        spans: List[SpanTuple],
        keep_longer: bool = True
    ) -> List[SpanTuple]:
        """
        Remove spans that are nested or overlapping.
        
        Args:
            spans: Input spans
            keep_longer: If True, keep longer spans in case of overlap
            
        Returns:
            Non-overlapping spans
        """
        if not spans:
            return []
        
        # Sort by start, then by length (descending if keep_longer)
        if keep_longer:
            sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
        else:
            sorted_spans = sorted(spans, key=lambda x: x[0])
        
        result = []
        covered_positions = set()
        
        for start, end, category in sorted_spans:
            # Check if any position is covered
            is_covered = any(pos in covered_positions for pos in range(start, end))
            
            if not is_covered:
                result.append((start, end, category))
                # Mark positions as covered
                for pos in range(start, end):
                    covered_positions.add(pos)
        
        return result


class MergeResolver:
    """High-level resolver combining multiple detection methods."""
    
    def __init__(
        self,
        strategy: str = "regex_priority",
        remove_nested: bool = True
    ):
        """
        Initialize resolver.
        
        Args:
            strategy: Merge strategy ('regex_priority', 'ner_priority', 'union')
            remove_nested: Whether to remove nested/overlapping spans
        """
        self.strategy = strategy
        self.remove_nested = remove_nested
    
    def merge(
        self,
        regex_spans: List[SpanTuple],
        ner_spans: List[SpanTuple],
        text: Optional[str] = None
    ) -> List[SpanTuple]:
        """
        Merge predictions from multiple detectors.
        
        Args:
            regex_spans: Regex detector output
            ner_spans: NER model output
            text: Optional original text for validation
            
        Returns:
            Merged spans
        """
        if self.strategy == "regex_priority":
            result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        elif self.strategy == "ner_priority":
            # Same logic but with NER taking priority
            result = ConflictResolver.resolve_conflicts(ner_spans, regex_spans)
        
        elif self.strategy == "union":
            # Take union, remove duplicates and overlaps
            all_spans = regex_spans + ner_spans
            all_spans = ConflictResolver._deduplicate_spans(all_spans)
            if self.remove_nested:
                result = ConflictResolver.remove_nested_overlaps(all_spans, keep_longer=True)
            else:
                result = all_spans
        
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using regex_priority")
            result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Final validation and sorting
        if text:
            result = self._validate_spans(result, text)
        
        result.sort(key=lambda x: x[0])
        
        return result
    
    @staticmethod
    def _validate_spans(
        spans: List[SpanTuple],
        text: str
    ) -> List[SpanTuple]:
        """Validate spans are within text bounds."""
        text_len = len(text)
        valid = []
        
        for start, end, category in spans:
            if 0 <= start < end <= text_len:
                span_text = text[start:end]
                if span_text.strip():
                    valid.append((start, end, category))
        
        return valid
