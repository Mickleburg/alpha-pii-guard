"""Utility functions for working with PII spans."""

from typing import List, Tuple, Optional, Set

SpanTuple = Tuple[int, int, str]


def sort_spans(spans: List[SpanTuple]) -> List[SpanTuple]:
    """
    Sort spans by start position (then by end position, then by category).
    
    Args:
        spans: List of (start, end, category) tuples
        
    Returns:
        Sorted list of spans
    """
    if not spans:
        return []
    return sorted(spans, key=lambda x: (x[0], x[1], x[2]))


def deduplicate_spans(spans: List[SpanTuple]) -> List[SpanTuple]:
    """
    Remove exact duplicate spans.
    
    Args:
        spans: List of spans
        
    Returns:
        List of unique spans (preserves first occurrence)
    """
    if not spans:
        return []
    
    seen = set()
    result = []
    
    for span in spans:
        if span not in seen:
            seen.add(span)
            result.append(span)
    
    return result


def spans_overlap(span_a: SpanTuple, span_b: SpanTuple) -> bool:
    """
    Check if two spans overlap.
    
    Spans overlap if one starts before the other ends.
    Adjacent spans (a.end == b.start) do NOT overlap.
    
    Args:
        span_a: First span (start, end, category)
        span_b: Second span (start, end, category)
        
    Returns:
        True if spans overlap
    """
    start_a, end_a, _ = span_a
    start_b, end_b, _ = span_b
    
    # No overlap if one ends at or before the other starts
    return not (end_a <= start_b or end_b <= start_a)


def span_contains(outer: SpanTuple, inner: SpanTuple) -> bool:
    """
    Check if outer span fully contains inner span.
    
    Args:
        outer: Potentially containing span
        inner: Potentially contained span
        
    Returns:
        True if outer contains inner (including edges)
    """
    start_outer, end_outer, _ = outer
    start_inner, end_inner, _ = inner
    
    return start_outer <= start_inner and end_inner <= end_outer


def spans_identical(span_a: SpanTuple, span_b: SpanTuple) -> bool:
    """
    Check if two spans are identical.
    
    Args:
        span_a: First span
        span_b: Second span
        
    Returns:
        True if start, end, and category match
    """
    start_a, end_a, cat_a = span_a
    start_b, end_b, cat_b = span_b
    
    return start_a == start_b and end_a == end_b and cat_a == cat_b


def is_valid_span(span: SpanTuple, text: Optional[str] = None) -> bool:
    """
    Validate a single span.
    
    Checks:
    - start < end (non-empty span)
    - start >= 0 (non-negative)
    - if text provided: end <= len(text)
    - span text is not pure whitespace
    
    Args:
        span: Span tuple to validate
        text: Optional text to validate span bounds against
        
    Returns:
        True if span is valid
    """
    try:
        start, end, category = span
    except (ValueError, TypeError):
        return False
    
    # Check basic properties
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    
    if start >= end or start < 0:
        return False
    
    if not isinstance(category, str) or not category.strip():
        return False
    
    # Check text bounds if provided
    if text is not None:
        if end > len(text):
            return False
        
        span_text = text[start:end]
        if not span_text.strip():
            return False
    
    return True


def normalize_spans(
    spans: List[SpanTuple],
    text: Optional[str] = None,
    drop_invalid: bool = True
) -> List[SpanTuple]:
    """
    Normalize spans: deduplicate, sort, validate.
    
    Args:
        spans: Input spans
        text: Optional text for validation
        drop_invalid: If True, remove invalid spans; if False, raise error
        
    Returns:
        Normalized spans
    """
    if not spans:
        return []
    
    # Deduplicate
    unique = deduplicate_spans(spans)
    
    # Validate
    if drop_invalid:
        valid = [s for s in unique if is_valid_span(s, text)]
    else:
        for s in unique:
            if not is_valid_span(s, text):
                raise ValueError(f"Invalid span: {s}")
        valid = unique
    
    # Sort
    return sort_spans(valid)


def get_covered_positions(spans: List[SpanTuple]) -> Set[int]:
    """
    Get all character positions covered by spans.
    
    Args:
        spans: List of spans
        
    Returns:
        Set of covered character positions
    """
    positions = set()
    for start, end, _ in spans:
        for pos in range(start, end):
            positions.add(pos)
    return positions


def remove_overlaps_with_priority(
    primary_spans: List[SpanTuple],
    secondary_spans: List[SpanTuple],
    text: Optional[str] = None
) -> List[SpanTuple]:
    """
    Merge two lists of spans, keeping primary spans and dropping overlapping secondary.
    
    This is the key merge logic:
    1. All primary spans are kept (highest priority)
    2. Secondary spans are added only if they don't overlap with any primary span
    3. Result is deduplicated and sorted
    
    Args:
        primary_spans: Spans with highest priority (typically regex)
        secondary_spans: Spans with lower priority (typically NER)
        text: Optional text for validation
        
    Returns:
        Merged list of non-overlapping spans
    """
    # Start with primary spans
    result = list(primary_spans)
    
    # Add secondary spans that don't conflict
    for secondary_span in secondary_spans:
        # Check for conflict with any primary span
        has_conflict = any(
            spans_overlap(secondary_span, primary_span)
            for primary_span in primary_spans
        )
        
        if not has_conflict:
            result.append(secondary_span)
    
    # Normalize result
    return normalize_spans(result, text=text, drop_invalid=True)


def merge_non_overlapping(
    spans_a: List[SpanTuple],
    spans_b: List[SpanTuple],
    text: Optional[str] = None
) -> List[SpanTuple]:
    """
    Merge two lists of spans, removing overlaps (keep longer spans).
    
    Unlike remove_overlaps_with_priority, this treats both lists equally
    and keeps longer spans in case of overlap.
    
    Args:
        spans_a: First list of spans
        spans_b: Second list of spans
        text: Optional text for validation
        
    Returns:
        Merged list with overlaps removed (keep longer)
    """
    if not spans_a and not spans_b:
        return []
    
    # Combine all spans
    all_spans = list(spans_a) + list(spans_b)
    
    # Deduplicate first
    unique = deduplicate_spans(all_spans)
    
    # Sort by start, then by length (descending) to keep longer spans
    sorted_spans = sorted(unique, key=lambda x: (x[0], -(x[1] - x[0])))
    
    # Remove overlaps by tracking covered positions
    result = []
    covered = set()
    
    for start, end, category in sorted_spans:
        # Check if any position is already covered
        is_covered = any(pos in covered for pos in range(start, end))
        
        if not is_covered:
            result.append((start, end, category))
            # Mark positions as covered
            for pos in range(start, end):
                covered.add(pos)
    
    # Normalize result
    return normalize_spans(result, text=text, drop_invalid=True)
