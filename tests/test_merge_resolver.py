"""
Tests for entity merge resolver.
"""

import pytest
from ml.entities import Entity
from ml.merge.resolver import (
    normalize_entities,
    deduplicate_entities,
    choose_best_entity,
    merge_entity_objects,
    merge_entities,
)


def test_deduplicate_exact_same():
    """Dedup removes exact duplicates."""
    entities = [
        Entity(0, 5, "Email", "regex"),
        Entity(0, 5, "Email", "regex"),
        Entity(0, 5, "Email", "ner"),
    ]
    result = deduplicate_entities(entities)
    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 5
    assert result[0].label == "Email"


def test_deduplicate_keeps_highest_score():
    """Dedup keeps entity with highest score."""
    entities = [
        Entity(0, 5, "Email", "regex", score=0.8),
        Entity(0, 5, "Email", "ner", score=0.95),
    ]
    result = deduplicate_entities(entities)
    assert len(result) == 1
    assert result[0].score == 0.95


def test_nested_same_label_prefers_shorter():
    """Nested same label: prefer more precise (shorter)."""
    a = Entity(0, 20, "ФИО", "ner", score=0.9)
    b = Entity(5, 15, "ФИО", "regex", score=0.8)
    
    best = choose_best_entity(a, b)
    assert best.start == 5
    assert best.end == 15


def test_overlap_different_labels_regex_priority():
    """Overlapping different labels: regex priority wins."""
    a = Entity(0, 10, "Email", "regex")
    b = Entity(5, 15, "ФИО", "ner")
    
    best = choose_best_entity(a, b)
    assert best.label == "Email"
    assert best.source == "regex"


def test_overlap_different_labels_ner_priority():
    """Overlapping different labels: NER priority wins."""
    a = Entity(0, 10, "ФИО", "ner")
    b = Entity(5, 15, "Email", "regex")
    
    best = choose_best_entity(a, b)
    assert best.label == "ФИО"
    assert best.source == "ner"


def test_non_overlapping_kept():
    """Non-overlapping entities both kept."""
    regex_entities = [Entity(0, 5, "Email", "regex")]
    ner_entities = [Entity(10, 20, "ФИО", "ner")]
    
    merged = merge_entity_objects(regex_entities, ner_entities)
    assert len(merged) == 2
    assert {e.label for e in merged} == {"Email", "ФИО"}


def test_deterministic_sorting():
    """Merged output is deterministically sorted."""
    regex_entities = [
        Entity(20, 30, "Email", "regex"),
        Entity(0, 5, "Номер телефона", "regex"),
    ]
    ner_entities = [
        Entity(10, 15, "ФИО", "ner"),
    ]
    
    merged = merge_entity_objects(regex_entities, ner_entities)
    starts = [e.start for e in merged]
    assert starts == sorted(starts)


def test_tuple_input_support():
    """merge_entities supports tuple input."""
    regex_spans = [(0, 10, "Email")]
    ner_spans = [(5, 15, "ФИО")]
    
    result = merge_entities(regex_spans, ner_spans)
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 3


def test_entity_input_support():
    """merge_entity_objects supports Entity input."""
    regex_entities = [Entity(0, 10, "Email", "regex")]
    ner_entities = [Entity(20, 30, "ФИО", "ner")]
    
    result = merge_entity_objects(regex_entities, ner_entities)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(e, Entity) for e in result)


def test_score_tiebreak():
    """Higher score wins in tie."""
    a = Entity(0, 10, "Email", "regex", score=0.7)
    b = Entity(0, 10, "Email", "ner", score=0.9)
    
    best = choose_best_entity(a, b)
    assert best.score == 0.9


def test_normalize_strips_whitespace():
    """Normalization trims whitespace from boundaries."""
    text = "  test@example.com  "
    entities = [Entity(0, len(text), "Email", "regex", text=text)]
    
    normalized = normalize_entities(entities, text)
    assert len(normalized) == 1
    assert normalized[0].start > 0
    assert normalized[0].end < len(text)
    assert normalized[0].text == "test@example.com"


def test_invalid_spans_filtered():
    """Invalid entities filtered out."""
    text = "Sample text"
    entities = [
        Entity(-1, 5, "Email", "regex"),  # Invalid start
        Entity(0, 100, "Email", "regex"),  # Out of bounds
        Entity(5, 3, "Email", "regex"),  # start >= end
        Entity(0, 5, "Email", "regex"),  # Valid
    ]
    
    normalized = normalize_entities(entities, text)
    assert len(normalized) == 1
    assert normalized[0].start == 0
    assert normalized[0].end == 5
