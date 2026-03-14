"""Tests for merge resolution logic."""

import pytest

from src.merge.resolver import ConflictResolver, MergeResolver

SpanTuple = tuple


class TestConflictDetection:
    """Test conflict detection between spans."""
    
    def test_no_overlap(self):
        """Test non-overlapping spans."""
        span1 = (0, 5, "A")
        span2 = (10, 15, "B")
        
        assert not ConflictResolver.spans_overlap(span1, span2)
        assert not ConflictResolver.spans_overlap(span2, span1)
    
    def test_adjacent_no_overlap(self):
        """Test adjacent spans (no overlap)."""
        span1 = (0, 5, "A")
        span2 = (5, 10, "B")
        
        # Adjacent but not overlapping
        assert not ConflictResolver.spans_overlap(span1, span2)
        assert not ConflictResolver.spans_overlap(span2, span1)
    
    def test_overlapping(self):
        """Test overlapping spans."""
        span1 = (0, 10, "A")
        span2 = (5, 15, "B")
        
        assert ConflictResolver.spans_overlap(span1, span2)
        assert ConflictResolver.spans_overlap(span2, span1)
    
    def test_one_contained_in_other(self):
        """Test span contained in another."""
        span1 = (0, 20, "A")
        span2 = (5, 15, "B")
        
        assert ConflictResolver.spans_overlap(span1, span2)
        assert ConflictResolver.spans_overlap(span2, span1)
    
    def test_identical_spans(self):
        """Test identical spans."""
        span1 = (0, 5, "A")
        span2 = (0, 5, "A")
        
        assert ConflictResolver.spans_identical(span1, span2)
        assert ConflictResolver.spans_overlap(span1, span2)
    
    def test_same_position_different_category(self):
        """Test same position but different category."""
        span1 = (0, 5, "A")
        span2 = (0, 5, "B")
        
        assert not ConflictResolver.spans_identical(span1, span2)
        assert ConflictResolver.spans_overlap(span1, span2)


class TestSpanContainment:
    """Test span containment."""
    
    def test_contains(self):
        """Test outer contains inner."""
        outer = (0, 20, "A")
        inner = (5, 15, "B")
        
        assert ConflictResolver.span_contains(outer, inner)
        assert not ConflictResolver.span_contains(inner, outer)
    
    def test_identical_not_contained(self):
        """Test identical spans."""
        span1 = (0, 5, "A")
        span2 = (0, 5, "B")
        
        # Identical positions but different categories
        assert ConflictResolver.span_contains(span1, span2)
        assert ConflictResolver.span_contains(span2, span1)
    
    def test_no_containment(self):
        """Test non-contained spans."""
        span1 = (0, 5, "A")
        span2 = (10, 15, "B")
        
        assert not ConflictResolver.span_contains(span1, span2)
        assert not ConflictResolver.span_contains(span2, span1)


class TestRegexPriority:
    """Test regex priority in conflict resolution."""
    
    def test_regex_wins_over_ner(self):
        """Test that regex spans are kept when conflicting with NER."""
        regex_spans = [(0, 10, "REGEX_MATCH")]
        ner_spans = [(5, 15, "NER_MATCH")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Should keep regex, drop NER (conflict)
        assert len(result) == 1
        assert (0, 10, "REGEX_MATCH") in result
        assert (5, 15, "NER_MATCH") not in result
    
    def test_non_overlapping_union(self):
        """Test that non-overlapping spans are both kept."""
        regex_spans = [(0, 5, "REGEX")]
        ner_spans = [(10, 15, "NER")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Both should be in result (no conflict)
        assert len(result) == 2
        assert (0, 5, "REGEX") in result
        assert (10, 15, "NER") in result
    
    def test_multiple_conflicts(self):
        """Test multiple conflicting spans."""
        regex_spans = [(0, 10, "A"), (20, 30, "B")]
        ner_spans = [(5, 15, "C"), (25, 35, "D")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Both regex spans should be kept
        assert (0, 10, "A") in result
        assert (20, 30, "B") in result
        # NER spans conflict with regex, should be dropped
        assert (5, 15, "C") not in result
        assert (25, 35, "D") not in result
    
    def test_regex_partial_overlap(self):
        """Test regex with partial overlap to NER."""
        regex_spans = [(0, 8, "REGEX")]
        ner_spans = [(5, 15, "NER")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Partial overlap - regex wins, NER dropped
        assert (0, 8, "REGEX") in result
        assert (5, 15, "NER") not in result


class TestDuplicateRemoval:
    """Test duplicate span removal."""
    
    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate spans."""
        spans = [(0, 5, "A"), (0, 5, "A"), (10, 15, "B")]
        
        result = ConflictResolver._deduplicate_spans(spans)
        
        assert len(result) == 2
        assert (0, 5, "A") in result
        assert (10, 15, "B") in result
    
    def test_keep_different_categories(self):
        """Test that spans with different categories are kept."""
        spans = [(0, 5, "A"), (0, 5, "B")]
        
        result = ConflictResolver._deduplicate_spans(spans)
        
        # Different categories - should keep both
        assert len(result) == 2
    
    def test_empty_list(self):
        """Test with empty list."""
        result = ConflictResolver._deduplicate_spans([])
        assert result == []


class TestNestedOverlapRemoval:
    """Test removal of nested overlaps."""
    
    def test_remove_contained_short(self):
        """Test removing span contained in longer span."""
        spans = [(0, 20, "A"), (5, 10, "B")]
        
        result = ConflictResolver.remove_nested_overlaps(spans, keep_longer=True)
        
        # Should keep only longer span
        assert len(result) == 1
        assert (0, 20, "A") in result
    
    def test_keep_longer_overlapping(self):
        """Test keeping longer of two overlapping spans."""
        spans = [(0, 10, "A"), (5, 15, "B")]
        
        result = ConflictResolver.remove_nested_overlaps(spans, keep_longer=True)
        
        # Both have same length overlap, should keep first (0, 10)
        assert len(result) == 1
        assert (0, 10, "A") in result
    
    def test_multiple_non_overlapping(self):
        """Test with multiple non-overlapping spans."""
        spans = [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")]
        
        result = ConflictResolver.remove_nested_overlaps(spans)
        
        # All non-overlapping, should keep all
        assert len(result) == 3
    
    def test_complex_overlap(self):
        """Test complex overlap scenario."""
        spans = [(0, 20, "A"), (5, 25, "B"), (30, 35, "C")]
        
        result = ConflictResolver.remove_nested_overlaps(spans, keep_longer=True)
        
        # (0, 20) and (5, 25) overlap - keep (5, 25) as longer
        # (30, 35) doesn't overlap
        assert len(result) == 2


class TestMergeResolver:
    """Test high-level merge resolver."""
    
    def test_regex_priority_strategy(self):
        """Test regex_priority merge strategy."""
        regex_spans = [(0, 10, "REGEX")]
        ner_spans = [(5, 15, "NER")]
        
        resolver = MergeResolver(strategy="regex_priority")
        result = resolver.merge(regex_spans, ner_spans)
        
        assert (0, 10, "REGEX") in result
        assert (5, 15, "NER") not in result
    
    def test_ner_priority_strategy(self):
        """Test ner_priority merge strategy."""
        regex_spans = [(0, 10, "REGEX")]
        ner_spans = [(5, 15, "NER")]
        
        resolver = MergeResolver(strategy="ner_priority")
        result = resolver.merge(regex_spans, ner_spans)
        
        # NER takes priority
        assert (0, 10, "REGEX") not in result
        assert (5, 15, "NER") in result
    
    def test_union_strategy(self):
        """Test union merge strategy."""
        regex_spans = [(0, 5, "REGEX")]
        ner_spans = [(10, 15, "NER")]
        
        resolver = MergeResolver(strategy="union")
        result = resolver.merge(regex_spans, ner_spans)
        
        # Both should be in result
        assert (0, 5, "REGEX") in result
        assert (10, 15, "NER") in result
    
    def test_sorted_output(self):
        """Test that output is sorted by start position."""
        regex_spans = [(20, 25, "A"), (0, 5, "B")]
        ner_spans = []
        
        resolver = MergeResolver()
        result = resolver.merge(regex_spans, ner_spans)
        
        # Should be sorted by start position
        assert result[0][0] == 0
        assert result[1][0] == 20


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        result = ConflictResolver.resolve_conflicts([], [])
        assert result == []
    
    def test_only_regex(self):
        """Test with only regex spans."""
        regex_spans = [(0, 5, "A"), (10, 15, "B")]
        ner_spans = []
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        assert len(result) == 2
        assert (0, 5, "A") in result
        assert (10, 15, "B") in result
    
    def test_only_ner(self):
        """Test with only NER spans."""
        regex_spans = []
        ner_spans = [(0, 5, "A"), (10, 15, "B")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        assert len(result) == 2
        assert (0, 5, "A") in result
        assert (10, 15, "B") in result
    
    def test_single_position_spans(self):
        """Test with spans of length 1 (edge case)."""
        regex_spans = [(0, 1, "A")]
        ner_spans = [(1, 2, "B")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Adjacent, not overlapping
        assert len(result) == 2
    
    def test_identical_regex_ner(self):
        """Test identical spans from regex and NER."""
        regex_spans = [(0, 5, "A")]
        ner_spans = [(0, 5, "A")]
        
        result = ConflictResolver.resolve_conflicts(regex_spans, ner_spans)
        
        # Should deduplicate to single span
        assert len(result) == 1
        assert (0, 5, "A") in result
    
    def test_many_overlapping_spans(self):
        """Test with many overlapping spans."""
        regex_spans = [(0, 10, "REGEX")] * 3  # Duplicate regex spans
        ner_spans = [(5, 15, "NER")] * 2  # Duplicate NER spans
        
        resolver = MergeResolver(strategy="regex_priority")
        result = resolver.merge(regex_spans, ner_spans)
        
        # Should deduplicate and keep only regex (priority)
        assert len(result) == 1
        assert (0, 10, "REGEX") in result


class TestValidation:
    """Test span validation."""
    
    def test_validate_valid_spans(self):
        """Test validation with valid spans."""
        spans = [(0, 5, "A"), (10, 15, "B")]
        text = "Hello world test"
        
        result = MergeResolver._validate_spans(spans, text)
        
        # Both spans valid
        assert len(result) == 2
    
    def test_validate_out_of_bounds(self):
        """Test validation with out-of-bounds spans."""
        spans = [(0, 5, "A"), (100, 200, "B")]
        text = "Hello"
        
        result = MergeResolver._validate_spans(spans, text)
        
        # First valid, second invalid
        assert len(result) == 1
        assert (0, 5, "A") in result
    
    def test_validate_whitespace_only(self):
        """Test validation of whitespace-only spans."""
        spans = [(0, 5, "A"), (5, 10, "B")]
        text = "Hello     "
        
        result = MergeResolver._validate_spans(spans, text)
        
        # First valid, second is whitespace
        assert len(result) == 1
        assert (0, 5, "A") in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
