"""Data preprocessing and cleaning utilities."""

import re
from pathlib import Path
from typing import List, Tuple, Optional

from src.utils.logging_utils import get_logger
from src.data.schemas import Document, Entity

logger = get_logger(__name__)


def validate_document(doc: Document) -> Tuple[bool, Optional[str]]:
    """
    Validate a document for data quality.
    
    Checks:
    - Text is not empty
    - All entity spans are within text bounds
    - Entity spans don't overlap
    - Entity categories are not empty
    
    Args:
        doc: Document to validate
        
    Returns:
        (is_valid, error_message)
    """
    # Check text
    if not doc.text or not doc.text.strip():
        return False, "Empty text"
    
    # Check each entity
    text_len = len(doc.text)
    for i, entity in enumerate(doc.entities):
        # Bounds check
        if entity.start < 0 or entity.end > text_len:
            return False, f"Entity {i}: span ({entity.start}, {entity.end}) out of bounds (text_len={text_len})"
        
        # Start < end check
        if entity.start >= entity.end:
            return False, f"Entity {i}: invalid span ({entity.start}, {entity.end})"
        
        # Category check
        if not entity.category or not entity.category.strip():
            return False, f"Entity {i}: empty category"
        
        # Check text at span is not all whitespace
        span_text = doc.text[entity.start:entity.end]
        if not span_text or not span_text.strip():
            return False, f"Entity {i}: span contains only whitespace"
    
    # Check for overlapping entities
    sorted_entities = sorted(doc.entities, key=lambda e: e.start)
    for i in range(len(sorted_entities) - 1):
        if sorted_entities[i].end > sorted_entities[i + 1].start:
            return False, f"Entities {i} and {i+1}: overlapping spans"
    
    return True, None


def normalize_whitespace(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Normalize multiple spaces to single space without changing entity coordinates.
    
    Returns both normalized text and position mapping.
    
    Args:
        text: Input text
        
    Returns:
        (normalized_text, position_map) where position_map[i] = original_position
    """
    # Track character mapping
    position_map = []
    normalized = []
    
    i = 0
    while i < len(text):
        if text[i].isspace():
            # Consume all consecutive whitespace
            normalized.append(" ")
            position_map.append(i)
            while i < len(text) and text[i].isspace():
                i += 1
        else:
            normalized.append(text[i])
            position_map.append(i)
            i += 1
    
    return "".join(normalized), position_map


def remap_entity_spans(
    entity: Entity,
    position_map: List[int]
) -> Optional[Entity]:
    """
    Remap entity spans to normalized text coordinates.
    
    Args:
        entity: Original entity
        position_map: Mapping from normalized to original positions
        
    Returns:
        Remapped entity or None if invalid
    """
    # Find start in normalized text
    if entity.start >= len(position_map):
        return None
    
    # Find end in normalized text
    if entity.end > len(position_map):
        return None
    
    return Entity(
        start=entity.start,
        end=entity.end,
        category=entity.category
    )


def preprocess_document(
    doc: Document,
    no_text_mutation: bool = True,
    validate: bool = True
) -> Optional[Document]:
    """
    Preprocess a single document.
    
    Args:
        doc: Input document
        no_text_mutation: If True, don't modify text (skip normalization)
        validate: Whether to validate document
        
    Returns:
        Preprocessed document or None if invalid
    """
    # Validate first
    if validate:
        is_valid, error_msg = validate_document(doc)
        if not is_valid:
            logger.warning(f"Invalid document {doc.doc_id}: {error_msg}")
            return None
    
    # By default, don't normalize (preserve exact char positions)
    if no_text_mutation:
        return doc
    
    # If normalization is enabled (usually disabled for production)
    # normalized_text, position_map = normalize_whitespace(doc.text)
    # normalized_entities = []
    # for entity in doc.entities:
    #     remapped = remap_entity_spans(entity, position_map)
    #     if remapped:
    #         normalized_entities.append(remapped)
    # 
    # return Document(
    #     text=normalized_text,
    #     entities=normalized_entities,
    #     doc_id=doc.doc_id
    # )
    
    return doc


def preprocess_documents(
    documents: List[Document],
    no_text_mutation: bool = True,
    validate: bool = True
) -> Tuple[List[Document], int]:
    """
    Preprocess multiple documents.
    
    Args:
        documents: List of input documents
        no_text_mutation: If True, don't modify text
        validate: Whether to validate documents
        
    Returns:
        (processed_documents, num_skipped)
    """
    processed = []
    skipped = 0
    
    for doc in documents:
        result = preprocess_document(doc, no_text_mutation=no_text_mutation, validate=validate)
        if result is not None:
            processed.append(result)
        else:
            skipped += 1
    
    logger.info(f"Preprocessed {len(processed)} documents (skipped {skipped})")
    return processed, skipped


def train_valid_split(
    documents: List[Document],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Document], List[Document]]:
    """
    Split documents into train and validation sets.
    
    Args:
        documents: List of documents
        train_ratio: Proportion for training (default 0.8)
        seed: Random seed
        
    Returns:
        (train_documents, valid_documents)
    """
    import random
    random.seed(seed)
    
    shuffled = list(documents)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    valid = shuffled[split_idx:]
    
    logger.info(f"Train/valid split: {len(train)} / {len(valid)}")
    return train, valid


def calculate_dataset_statistics(documents: List[Document]) -> dict:
    """
    Calculate statistics about the dataset.
    
    Args:
        documents: List of documents
        
    Returns:
        Dictionary with statistics
    """
    total_docs = len(documents)
    total_entities = sum(len(doc.entities) for doc in documents)
    total_chars = sum(len(doc.text) for doc in documents)
    avg_entities_per_doc = total_entities / total_docs if total_docs > 0 else 0
    avg_chars_per_doc = total_chars / total_docs if total_docs > 0 else 0
    
    # Category distribution
    category_counts = {}
    for doc in documents:
        for entity in doc.entities:
            category_counts[entity.category] = category_counts.get(entity.category, 0) + 1
    
    # Span length distribution
    span_lengths = []
    for doc in documents:
        for entity in doc.entities:
            span_lengths.append(entity.end - entity.start)
    
    avg_span_length = sum(span_lengths) / len(span_lengths) if span_lengths else 0
    min_span_length = min(span_lengths) if span_lengths else 0
    max_span_length = max(span_lengths) if span_lengths else 0
    
    return {
        "total_documents": total_docs,
        "total_entities": total_entities,
        "total_characters": total_chars,
        "avg_entities_per_document": avg_entities_per_doc,
        "avg_characters_per_document": avg_chars_per_doc,
        "category_distribution": category_counts,
        "avg_span_length": avg_span_length,
        "min_span_length": min_span_length,
        "max_span_length": max_span_length,
        "num_categories": len(category_counts)
    }


def print_dataset_statistics(stats: dict) -> None:
    """Print dataset statistics in human-readable format."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Avg entities per document: {stats['avg_entities_per_document']:.2f}")
    print(f"Avg characters per document: {stats['avg_characters_per_document']:.2f}")
    print(f"Average span length: {stats['avg_span_length']:.2f}")
    print(f"Min/Max span length: {stats['min_span_length']} / {stats['max_span_length']}")
    print(f"Number of categories: {stats['num_categories']}")
    print("\nCategory distribution:")
    for category, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_entities']
        print(f"  {category}: {count} ({pct:.1f}%)")
    print("="*60 + "\n")
