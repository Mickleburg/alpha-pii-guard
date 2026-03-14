"""
Entity merge resolution logic for alpha-pii-guard.

Implements intelligent merging of regex and NER predictions with priority-based resolution.
"""

from typing import List, Tuple, Optional

from ml.entities import (
    Entity,
    entity_to_tuple,
    tuple_to_entity,
    validate_entity,
    overlaps,
    contains,
    exact_match,
    span_length,
)
from ml.merge.config import (
    get_category_priority,
    is_structured_label,
    DEFAULT_SOURCE_PRIORITY,
)


def normalize_entities(entities: List[Entity], text: Optional[str] = None) -> List[Entity]:
    """
    Normalize and validate entities.
    
    Filters out invalid entities and trims whitespace from boundaries.
    
    Args:
        entities: List of entities to normalize
        text: Optional source text for validation and trimming
        
    Returns:
        List of validated and normalized entities
    """
    normalized = []
    
    for entity in entities:
        # Validate
        if not validate_entity(entity, text):
            continue
        
        # Trim whitespace if text available
        if text is not None:
            start, end = entity.start, entity.end
            span_text = text[start:end]
            
            # Strip leading whitespace
            stripped = span_text.lstrip()
            start += len(span_text) - len(stripped)
            
            # Strip trailing whitespace
            stripped = stripped.rstrip()
            end = start + len(stripped)
            
            # Skip if empty after trimming
            if start >= end:
                continue
            
            entity = Entity(
                start=start,
                end=end,
                label=entity.label,
                source=entity.source,
                score=entity.score,
                text=stripped
            )
        
        normalized.append(entity)
    
    return normalized


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Remove exact duplicate entities.
    
    Keeps entity with highest score if duplicates exist.
    
    Args:
        entities: List of entities
        
    Returns:
        Deduplicated list
    """
    if not entities:
        return []
    
    # Group by (start, end, label)
    groups = {}
    for entity in entities:
        key = (entity.start, entity.end, entity.label)
        if key not in groups:
            groups[key] = []
        groups[key].append(entity)
    
    # Keep best from each group
    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Prefer entity with score, then highest score
            with_score = [e for e in group if e.score is not None]
            if with_score:
                best = max(with_score, key=lambda e: e.score)
            else:
                best = group[0]
            result.append(best)
    
    return result


def choose_best_entity(a: Entity, b: Entity, text: Optional[str] = None) -> Entity:
    """
    Choose best entity when conflict exists.
    
    Priority order:
    1. Same label + nested -> prefer more precise (exact match or shorter)
    2. Different labels -> category priority
    3. Same priority + same label -> prefer higher score
    4. Same score -> prefer structured=shorter, context=longer
    5. Fallback -> source priority (regex > ner)
    
    Args:
        a: First entity
        b: Second entity
        text: Optional source text
        
    Returns:
        Chosen entity
    """
    # Case 1: Exact match - arbitrary but deterministic
    if exact_match(a, b):
        return a if a.source <= b.source else b
    
    # Case 2: Same label, nested spans
    if a.label == b.label:
        # One contains the other
        if contains(a, b):
            return b  # Prefer more precise (nested)
        if contains(b, a):
            return a  # Prefer more precise (nested)
        
        # Overlapping same label -> prefer shorter for precision
        len_a = span_length(a)
        len_b = span_length(b)
        if len_a != len_b:
            return a if len_a < len_b else b
        
        # Equal length, use score
        if a.score is not None and b.score is not None:
            if a.score != b.score:
                return a if a.score > b.score else b
        elif a.score is not None:
            return a
        elif b.score is not None:
            return b
        
        # Fallback: source priority
        return a if DEFAULT_SOURCE_PRIORITY.get(a.source, 999) < DEFAULT_SOURCE_PRIORITY.get(b.source, 999) else b
    
    # Case 3: Different labels
    priority_a = get_category_priority(a.label)
    priority_b = get_category_priority(b.label)
    
    # Check if category priorities differ
    if priority_a == "regex" and b.source == "ner":
        return a
    if priority_b == "regex" and a.source == "ner":
        return b
    if priority_a == "ner" and b.source == "regex":
        return b
    if priority_b == "ner" and a.source == "regex":
        return a
    
    # Same category priority, use scores
    if a.score is not None and b.score is not None:
        if a.score != b.score:
            return a if a.score > b.score else b
    elif a.score is not None:
        return a
    elif b.score is not None:
        return b
    
    # Tie-break by span length based on label type
    len_a = span_length(a)
    len_b = span_length(b)
    
    if is_structured_label(a.label) or is_structured_label(b.label):
        # Prefer shorter for structured
        if len_a != len_b:
            return a if len_a < len_b else b
    else:
        # Prefer longer for context
        if len_a != len_b:
            return a if len_a > len_b else b
    
    # Final fallback: source priority
    return a if DEFAULT_SOURCE_PRIORITY.get(a.source, 999) < DEFAULT_SOURCE_PRIORITY.get(b.source, 999) else b


def merge_entity_objects(
    regex_entities: List[Entity],
    ner_entities: List[Entity],
    text: Optional[str] = None
) -> List[Entity]:
    """
    Merge regex and NER entities with intelligent conflict resolution.
    
    Args:
        regex_entities: Entities from regex detector
        ner_entities: Entities from NER detector
        text: Optional source text for validation
        
    Returns:
        Merged list of entities with source='merged'
    """
    # Normalize inputs
    regex_entities = normalize_entities(regex_entities, text)
    ner_entities = normalize_entities(ner_entities, text)
    
    all_entities = regex_entities + ner_entities
    
    if not all_entities:
        return []
    
    # Deduplicate exact matches
    all_entities = deduplicate_entities(all_entities)
    
    # Sort by start position for processing
    all_entities.sort(key=lambda e: (e.start, e.end, e.label))
    
    # Resolve overlaps
    merged = []
    i = 0
    
    while i < len(all_entities):
        current = all_entities[i]
        
        # Find all overlapping entities
        overlapping = [current]
        j = i + 1
        while j < len(all_entities) and overlaps(current, all_entities[j]):
            overlapping.append(all_entities[j])
            j += 1
        
        if len(overlapping) == 1:
            # No conflicts
            merged.append(Entity(
                start=current.start,
                end=current.end,
                label=current.label,
                source="merged",
                score=current.score,
                text=current.text
            ))
            i += 1
        else:
            # Resolve conflict
            best = overlapping[0]
            for candidate in overlapping[1:]:
                best = choose_best_entity(best, candidate, text)
            
            merged.append(Entity(
                start=best.start,
                end=best.end,
                label=best.label,
                source="merged",
                score=best.score,
                text=best.text
            ))
            
            # Skip processed entities
            i = j
    
    # Final sort for deterministic output
    merged.sort(key=lambda e: (e.start, e.end, e.label))
    
    return merged


def merge_entities(
    regex_spans: List[Tuple[int, int, str]],
    ner_spans: List[Tuple[int, int, str]],
    text: Optional[str] = None
) -> List[Tuple[int, int, str]]:
    """
    Merge regex and NER span tuples.
    
    Public API function using tuple format.
    
    Args:
        regex_spans: List of (start, end, label) from regex
        ner_spans: List of (start, end, label) from NER
        text: Optional source text
        
    Returns:
        Merged list of (start, end, label) tuples
    """
    # Convert to entities
    regex_entities = [tuple_to_entity(s, "regex") for s in regex_spans]
    ner_entities = [tuple_to_entity(s, "ner") for s in ner_spans]
    
    # Merge
    merged = merge_entity_objects(regex_entities, ner_entities, text)
    
    # Convert back to tuples
    return [entity_to_tuple(e) for e in merged]
