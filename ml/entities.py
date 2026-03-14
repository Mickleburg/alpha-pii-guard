"""
Unified entity representation and adapters for alpha-pii-guard.

Provides Entity dataclass and conversion utilities for regex, NER, and merged detectors.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

#from ml.merge.label_map import normalize_label
from ml.config.labels import normalize_label, strip_bio


@dataclass
class Entity:
    """
    Unified PII entity representation.
    
    Attributes:
        start: Character offset where entity begins (inclusive)
        end: Character offset where entity ends (exclusive)
        label: Normalized entity category name (without BIO prefixes)
        source: Detector source ('regex', 'ner', 'merged')
        score: Optional confidence score (0.0-1.0)
        text: Optional extracted text span
    """
    start: int
    end: int
    label: str
    source: str
    score: Optional[float] = None
    text: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and normalize entity after construction."""
        # Normalize label on construction
        #self.label = normalize_label(self.label)
        self.label = strip_bio(self.label)


def entity_to_tuple(entity: Entity) -> Tuple[int, int, str]:
    """
    Convert Entity to public tuple format.
    
    Args:
        entity: Entity instance
        
    Returns:
        Tuple of (start, end, label)
    """
    return (entity.start, entity.end, entity.label)


def tuple_to_entity(
    span: Tuple[int, int, str],
    source: str,
    score: Optional[float] = None,
    text: Optional[str] = None
) -> Entity:
    """
    Convert tuple format to Entity.
    
    Args:
        span: Tuple of (start, end, label)
        source: Detector source identifier
        score: Optional confidence score
        text: Optional extracted text
        
    Returns:
        Entity instance
    """
    start, end, label = span
    return Entity(
        start=start,
        end=end,
        label=normalize_label(label),
        source=source,
        score=score,
        text=text
    )


def validate_entity(entity: Entity, text: Optional[str] = None) -> bool:
    """
    Validate entity boundaries and structure.
    
    Args:
        entity: Entity to validate
        text: Optional source text for boundary checking
        
    Returns:
        True if entity is valid
    """
    # Check basic structure
    if entity.start < 0 or entity.end < 0:
        return False
    
    if entity.start >= entity.end:
        return False
    
    if not entity.label or not entity.label.strip():
        return False
    
    # Check text bounds if provided
    if text is not None:
        if entity.start >= len(text) or entity.end > len(text):
            return False
    
    return True


def overlaps(a: Entity, b: Entity) -> bool:
    """
    Check if two entities overlap.
    
    Args:
        a: First entity
        b: Second entity
        
    Returns:
        True if entities overlap
    """
    return not (a.end <= b.start or b.end <= a.start)


def contains(a: Entity, b: Entity) -> bool:
    """
    Check if entity a fully contains entity b.
    
    Args:
        a: Container entity
        b: Potentially contained entity
        
    Returns:
        True if a contains b
    """
    return a.start <= b.start and a.end >= b.end


def exact_match(a: Entity, b: Entity) -> bool:
    """
    Check if two entities have exact same span and label.
    
    Args:
        a: First entity
        b: Second entity
        
    Returns:
        True if entities match exactly
    """
    return a.start == b.start and a.end == b.end and a.label == b.label


def span_length(entity: Entity) -> int:
    """
    Calculate entity span length.
    
    Args:
        entity: Entity instance
        
    Returns:
        Length in characters
    """
    return entity.end - entity.start
