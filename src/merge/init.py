"""Merge layer for combining regex and NER predictions."""

from src.merge.merger import Merger
from src.merge.span_utils import (
    merge_overlapping_spans,
    resolve_conflicts,
    deduplicate_spans,
)

__all__ = [
    "Merger",
    "merge_overlapping_spans",
    "resolve_conflicts",
    "deduplicate_spans",
]
