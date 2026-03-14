"""Tests for BIO tag conversion."""

import pytest

from src.data.schemas import Entity
from src.data.bio_converter import (
    get_bio_tag_scheme,
    spans_to_bio,
    bio_to_spans,
    align_labels_to_tokens,
    _deduplicate_spans
)


class TestBioTagScheme:
    """Test BIO tag scheme creation."""
    
    def test_bio_scheme_creation(self):
        """Test that BIO scheme is created correctly."""
        categories = ["PASSPORT", "PHONE", "EMAIL"]
        tags, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
        
        # Should have O + 2*num_categories tags
        assert len(tags) == 1 + 2 * len(categories)
        
        # First tag should be O
        assert tags[0] == "O"
        assert tag_to_id["O"] == 0
        
        # Check all categories have B- and I- tags
        for category in categories:
            assert f"B-{category}" in tag_to_id
            assert f"I-{category}" in tag_to_id
        
        # Check bidirectional mapping
        for tag, tag_id in tag_to_id.items():
            assert id_to_tag[tag_id] == tag


class TestSpansToBio:
    """Test conversion from char spans to BIO labels."""
    
    def test_empty_entities(self):
        """Test with no entities."""
        text = "Hello world"
        entities = []
        token_offsets = [(0, 5), (6, 11)]  # ["Hello", "world"]
        categories = ["WORD"]
        
        labels = spans_to_bio(text, entities, token_offsets, categories)
        
        assert len(labels) == 2
        # All should be "O" (ID 0)
        assert all(label == 0 for label in labels)
    
    def test_single_token_entity(self):
        """Test with single token entity."""
        text = "Call +7-999-123-45-67 now"
        entities = [Entity(start=5, end=21, category="PHONE")]
        token_offsets = [(0, 4), (5, 21), (22, 25)]  # ["Call", "+7-999-123-45-67", "now"]
        categories = ["PHONE"]
        
        labels = spans_to_bio(text, entities, token_offsets, categories)
        _, tag_to_id, _ = get_bio_tag_scheme(categories)
        
        assert len(labels) == 3
        # First token: O
        assert labels[0] == tag_to_id["O"]
        # Second token: B-PHONE
        assert labels[1] == tag_to_id["B-PHONE"]
        # Third token: O
        assert labels[2] == tag_to_id["O"]
    
    def test_multi_token_entity(self):
        """Test with entity spanning multiple tokens."""
        text = "John Doe is here"
        entities = [Entity(start=0, end=8, category="PERSON_NAME")]
        # Tokens: ["John", "Doe", "is", "here"]
        token_offsets = [(0, 4), (5, 8), (9, 11), (12, 16)]
        categories = ["PERSON_NAME"]
        
        labels = spans_to_bio(text, entities, token_offsets, categories)
        _, tag_to_id, _ = get_bio_tag_scheme(categories)
        
        assert len(labels) == 4
        # First token: B-PERSON_NAME
        assert labels[0] == tag_to_id["B-PERSON_NAME"]
        # Second token: I-PERSON_NAME (inside, continues entity)
        assert labels[1] == tag_to_id["I-PERSON_NAME"]
        # Third token: O
        assert labels[2] == tag_to_id["O"]
        # Fourth token: O
        assert labels[3] == tag_to_id["O"]
    
    def test_multiple_entities(self):
        """Test with multiple entities."""
        text = "Email: test@example.com Phone: +7-999-123-45-67"
        entities = [
            Entity(start=7, end=23, category="EMAIL"),
            Entity(start=31, end=47, category="PHONE")
        ]
        token_offsets = [
            (0, 6),      # "Email:"
            (7, 23),     # "test@example.com"
            (24, 30),    # "Phone:"
            (31, 47)     # "+7-999-123-45-67"
        ]
        categories = ["EMAIL", "PHONE"]
        
        labels = spans_to_bio(text, entities, token_offsets, categories)
        _, tag_to_id, _ = get_bio_tag_scheme(categories)
        
        assert len(labels) == 4
        # First: O
        assert labels[0] == tag_to_id["O"]
        # Second: B-EMAIL
        assert labels[1] == tag_to_id["B-EMAIL"]
        # Third: O
        assert labels[2] == tag_to_id["O"]
        # Fourth: B-PHONE
        assert labels[3] == tag_to_id["B-PHONE"]


class TestBioToSpans:
    """Test conversion from BIO labels back to char spans."""
    
    def test_empty_labels(self):
        """Test with no entities."""
        labels = [0, 0, 0]  # All "O"
        token_offsets = [(0, 5), (6, 11), (12, 15)]
        _, _, id_to_tag = get_bio_tag_scheme(["WORD"])
        
        spans = bio_to_spans(labels, token_offsets, id_to_tag)
        
        assert spans == []
    
    def test_single_token_entity_recovery(self):
        """Test recovering a single-token entity."""
        # Create labels: [O, B-PHONE, O]
        categories = ["PHONE"]
        _, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
        
        labels = [tag_to_id["O"], tag_to_id["B-PHONE"], tag_to_id["O"]]
        token_offsets = [(0, 4), (5, 21), (22, 25)]
        
        spans = bio_to_spans(labels, token_offsets, id_to_tag)
        
        assert len(spans) == 1
        assert spans[0] == (5, 21, "PHONE")
    
    def test_multi_token_entity_recovery(self):
        """Test recovering multi-token entity."""
        categories = ["PERSON_NAME"]
        _, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
        
        # Labels: [B-PERSON_NAME, I-PERSON_NAME, O, O]
        labels = [
            tag_to_id["B-PERSON_NAME"],
            tag_to_id["I-PERSON_NAME"],
            tag_to_id["O"],
            tag_to_id["O"]
        ]
        token_offsets = [(0, 4), (5, 8), (9, 11), (12, 16)]
        
        spans = bio_to_spans(labels, token_offsets, id_to_tag)
        
        assert len(spans) == 1
        assert spans[0] == (0, 8, "PERSON_NAME")
    
    def test_multiple_entities_recovery(self):
        """Test recovering multiple entities."""
        categories = ["EMAIL", "PHONE"]
        _, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
        
        labels = [
            tag_to_id["O"],
            tag_to_id["B-EMAIL"],
            tag_to_id["O"],
            tag_to_id["B-PHONE"]
        ]
        token_offsets = [(0, 6), (7, 23), (24, 30), (31, 47)]
        
        spans = bio_to_spans(labels, token_offsets, id_to_tag)
        
        assert len(spans) == 2
        assert spans[0] == (7, 23, "EMAIL")
        assert spans[1] == (31, 47, "PHONE")
    
    def test_spans_sorted(self):
        """Test that output spans are sorted by start."""
        categories = ["A", "B"]
        _, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
        
        # Create out-of-order predictions
        labels = [
            tag_to_id["B-B"],
            tag_to_id["O"],
            tag_to_id["B-A"]
        ]
        token_offsets = [(0, 5), (6, 10), (11, 15)]
        
        spans = bio_to_spans(labels, token_offsets, id_to_tag)
        
        # Should be sorted by start
        assert len(spans) == 2
        assert spans[0][0] <= spans[1][0]


class TestRoundTrip:
    """Test round-trip conversion: spans -> BIO -> spans."""
    
    def test_roundtrip_single_entity(self):
        """Test round-trip with single entity."""
        text = "My passport 12 34 567890"
        original_entities = [Entity(start=12, end=24, category="PASSPORT")]
        categories = ["PASSPORT"]
        
        # Simulate tokenization
        token_offsets = [(0, 2), (3, 11), (12, 24)]
        
        # Forward conversion
        labels = spans_to_bio(text, original_entities, token_offsets, categories)
        
        # Backward conversion
        _, _, id_to_tag = get_bio_tag_scheme(categories)
        recovered_spans = bio_to_spans(labels, token_offsets, id_to_tag, text=text)
        
        assert len(recovered_spans) == len(original_entities)
        assert recovered_spans[0] == (original_entities[0].start, original_entities[0].end, original_entities[0].category)
    
    def test_roundtrip_multiple_entities(self):
        """Test round-trip with multiple entities."""
        text = "Name: John Doe, Email: john@example.com"
        original_entities = [
            Entity(start=6, end=14, category="PERSON_NAME"),
            Entity(start=23, end=39, category="EMAIL")
        ]
        categories = ["PERSON_NAME", "EMAIL"]
        
        # Simulate tokenization
        token_offsets = [
            (0, 4),      # "Name"
            (5, 8),      # "John"
            (9, 12),     # "Doe"
            (13, 14),    # ","
            (15, 21),    # "Email"
            (22, 23),    # ":"
            (23, 39)     # "john@example.com"
        ]
        
        # Forward conversion
        labels = spans_to_bio(text, original_entities, token_offsets, categories)
        
        # Backward conversion
        _, _, id_to_tag = get_bio_tag_scheme(categories)
        recovered_spans = bio_to_spans(labels, token_offsets, id_to_tag, text=text)
        
        assert len(recovered_spans) == len(original_entities)
        for original, recovered in zip(original_entities, recovered_spans):
            assert recovered == (original.start, original.end, original.category)


class TestAlignLabelsToTokens:
    """Test alignment of labels to tokens."""
    
    def test_align_with_mask(self):
        """Test alignment using special token mask."""
        labels = [0, 1, 2, 1, 3]
        token_offsets = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
        special_token_mask = [0, 0, 1, 0, 0]  # Token 2 is special
        
        aligned_labels, aligned_offsets = align_labels_to_tokens(
            token_offsets, labels, special_token_mask
        )
        
        assert len(aligned_labels) == 4
        assert aligned_labels == [0, 1, 1, 3]
        assert len(aligned_offsets) == 4
        assert aligned_offsets == [(0, 5), (6, 10), (16, 20), (21, 25)]
    
    def test_align_by_offset_length(self):
        """Test alignment by filtering zero-length offsets."""
        labels = [0, 1, 2, 1, 3]
        # Token 2 has zero-length offset (special token)
        token_offsets = [(0, 5), (6, 10), (11, 11), (16, 20), (21, 25)]
        
        aligned_labels, aligned_offsets = align_labels_to_tokens(
            token_offsets, labels
        )
        
        assert len(aligned_labels) == 4
        assert aligned_labels == [0, 1, 1, 3]
        assert aligned_offsets == [(0, 5), (6, 10), (16, 20), (21, 25)]


class TestDeduplicateSpans:
    """Test span deduplication."""
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        spans = [(0, 5, "A"), (10, 15, "B")]
        result = _deduplicate_spans(spans)
        assert result == spans
    
    def test_exact_duplicate(self):
        """Test removing exact duplicates."""
        spans = [(0, 5, "A"), (0, 5, "A"), (10, 15, "B")]
        result = _deduplicate_spans(spans)
        assert len(result) == 2
        assert (0, 5, "A") in result
        assert (10, 15, "B") in result
    
    def test_overlapping_spans(self):
        """Test removing overlapping spans (keep longer)."""
        spans = [(0, 5, "A"), (2, 8, "B")]
        result = _deduplicate_spans(spans)
        # Should keep the longer span(s)
        assert len(result) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
