"""BIO tagging and conversion utilities."""

import re
from typing import List, Tuple, Optional, Dict, Set

from src.utils.logging_utils import get_logger
from src.data.schemas import Entity, TokenizedDocument

logger = get_logger(__name__)

# BIO tag constants
BIO_TAGS = ["O"]  # Will be extended with B-*, I-* tags per category


def get_bio_tag_scheme(categories: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Create BIO tag scheme from category list.
    
    Tags: O, B-CATEGORY, I-CATEGORY for each category
    
    Args:
        categories: List of entity categories
        
    Returns:
        (tag_list, tag_to_id, id_to_tag)
    """
    tags = ["O"]
    
    for category in sorted(categories):
        tags.append(f"B-{category}")
        tags.append(f"I-{category}")
    
    tag_to_id = {tag: idx for idx, tag in enumerate(tags)}
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    
    return tags, tag_to_id, id_to_tag


def spans_to_bio(
    text: str,
    entities: List[Entity],
    token_offsets: List[Tuple[int, int]],
    categories: List[str]
) -> List[int]:
    """
    Convert character-level entity spans to token-level BIO labels.
    
    Args:
        text: Input text
        entities: List of Entity objects with char-level spans
        token_offsets: List of (start, end) offsets for each token
        categories: List of valid categories
        
    Returns:
        List of BIO label IDs (one per token, excluding special tokens)
    """
    # Create BIO tag scheme
    _, tag_to_id, _ = get_bio_tag_scheme(categories)
    
    # Initialize labels (all tokens start as "O")
    labels = ["O"] * len(token_offsets)
    
    # For each entity, assign B- and I- tags to overlapping tokens
    for entity in entities:
        if entity.category not in categories:
            logger.warning(f"Unknown category: {entity.category}")
            continue
        
        entity_start, entity_end = entity.start, entity.end
        first_token = True
        
        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            # Check if token overlaps with entity
            if token_start < entity_end and token_end > entity_start:
                # Token overlaps with entity
                if first_token:
                    labels[token_idx] = f"B-{entity.category}"
                    first_token = False
                else:
                    labels[token_idx] = f"I-{entity.category}"
    
    # Convert to IDs
    label_ids = [tag_to_id[label] for label in labels]
    return label_ids


def bio_to_spans(
    predicted_labels: List[int],
    token_offsets: List[Tuple[int, int]],
    id_to_tag: Dict[int, str],
    text: Optional[str] = None,
    threshold: float = 0.0
) -> List[Tuple[int, int, str]]:
    """
    Convert token-level BIO labels back to character-level entity spans.
    
    Args:
        predicted_labels: List of predicted BIO label IDs
        token_offsets: List of (start, end) offsets for each token
        id_to_tag: Mapping from label ID to tag string
        text: Optional text for validation
        threshold: Confidence threshold (if scores provided, unused here)
        
    Returns:
        List of (start, end, category) tuples
    """
    if not predicted_labels or not token_offsets:
        return []
    
    if len(predicted_labels) != len(token_offsets):
        logger.warning(
            f"Mismatch: {len(predicted_labels)} labels vs {len(token_offsets)} offsets"
        )
        return []
    
    spans = []
    current_entity_start = None
    current_entity_category = None
    
    for token_idx, label_id in enumerate(predicted_labels):
        tag = id_to_tag.get(label_id, "O")
        token_start, token_end = token_offsets[token_idx]
        
        # Parse tag
        if tag == "O":
            # Outside tag - close any open entity
            if current_entity_start is not None:
                spans.append((
                    current_entity_start,
                    token_offsets[token_idx - 1][1],  # End of previous token
                    current_entity_category
                ))
                current_entity_start = None
                current_entity_category = None
        
        elif tag.startswith("B-"):
            # Beginning tag
            # Close any open entity
            if current_entity_start is not None:
                spans.append((
                    current_entity_start,
                    token_offsets[token_idx - 1][1],  # End of previous token
                    current_entity_category
                ))
            
            # Start new entity
            category = tag[2:]  # Remove "B-" prefix
            current_entity_start = token_start
            current_entity_category = category
        
        elif tag.startswith("I-"):
            # Inside tag
            category = tag[2:]  # Remove "I-" prefix
            
            if current_entity_start is None:
                # I- tag without preceding B- tag (treat as B-)
                current_entity_start = token_start
                current_entity_category = category
            elif current_entity_category != category:
                # Category mismatch - close previous, start new
                spans.append((
                    current_entity_start,
                    token_offsets[token_idx - 1][1],
                    current_entity_category
                ))
                current_entity_start = token_start
                current_entity_category = category
    
    # Close any remaining open entity
    if current_entity_start is not None:
        spans.append((
            current_entity_start,
            token_offsets[-1][1],  # End of last token
            current_entity_category
        ))
    
    # Sort by start position
    spans = sorted(spans, key=lambda x: x[0])
    
    # Remove duplicates and overlaps
    spans = _deduplicate_spans(spans)
    
    # Validate if text provided
    if text:
        spans = _validate_spans(spans, text)
    
    return spans


def _deduplicate_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove duplicate and overlapping spans, keeping longer ones."""
    if not spans:
        return []
    
    # Sort by start, then by length (descending)
    sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    
    result = []
    covered = set()
    
    for start, end, category in sorted_spans:
        # Check if this span is covered by an existing one
        is_covered = False
        for char_pos in range(start, end):
            if char_pos in covered:
                is_covered = True
                break
        
        if not is_covered:
            result.append((start, end, category))
            for char_pos in range(start, end):
                covered.add(char_pos)
    
    return result


def _validate_spans(spans: List[Tuple[int, int, str]], text: str) -> List[Tuple[int, int, str]]:
    """Validate spans are within text bounds and remove invalid ones."""
    text_len = len(text)
    valid = []
    
    for start, end, category in spans:
        if 0 <= start < end <= text_len:
            valid.append((start, end, category))
        else:
            logger.warning(f"Invalid span ({start}, {end}) for text length {text_len}")
    
    return valid


def align_labels_to_tokens(
    token_offsets: List[Tuple[int, int]],
    labels: List[int],
    special_token_mask: Optional[List[int]] = None
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Filter out special tokens from labels and offsets.
    
    Args:
        token_offsets: Original token offsets
        labels: Original labels (one per token)
        special_token_mask: Mask where 1 = special token, 0 = real token
                           If None, filters tokens with zero-length offset
        
    Returns:
        (filtered_labels, filtered_offsets)
    """
    if special_token_mask is not None:
        # Use provided mask
        filtered_labels = [
            label for label, is_special in zip(labels, special_token_mask)
            if is_special == 0
        ]
        filtered_offsets = [
            offset for offset, is_special in zip(token_offsets, special_token_mask)
            if is_special == 0
        ]
    else:
        # Filter by zero-length offsets (subword tokens)
        filtered_labels = [
            label for label, (start, end) in zip(labels, token_offsets)
            if start != end  # Non-zero length
        ]
        filtered_offsets = [
            offset for offset in token_offsets
            if offset[0] != offset[1]  # Non-zero length
        ]
    
    return filtered_labels, filtered_offsets
