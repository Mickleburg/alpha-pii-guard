"""Input/Output operations for data loading and saving."""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

from src.utils.types import Span, SpanTuple, TrainSample, EntityModel, DocumentPrediction
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_train_dataset(path: str | Path) -> List[TrainSample]:
    """
    Load training dataset from TSV file.
    
    Format:
        text    entities
        "Sample text"    [{"start": 0, "end": 5, "category": "LABEL"}]
    
    Args:
        path: Path to TSV file
        
    Returns:
        List of TrainSample objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Train dataset not found: {path}")
    
    samples = []
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        
        if "text" not in df.columns:
            raise ValueError("Missing 'text' column in train dataset")
        if "entities" not in df.columns:
            raise ValueError("Missing 'entities' column in train dataset")
        
        for idx, row in df.iterrows():
            text = row["text"]
            entities_str = row["entities"]
            
            entities = []
            if pd.notna(entities_str) and entities_str.strip():
                try:
                    entities_list = json.loads(entities_str)
                    for entity_dict in entities_list:
                        entity = EntityModel(**entity_dict)
                        entities.append(entity)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse entities at row {idx}: {e}")
                    continue
            
            doc_id = row.get("id", str(idx))
            sample = TrainSample(text=text, entities=entities, doc_id=doc_id)
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} training samples from {path}")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading train dataset from {path}: {e}")
        raise


def load_test_dataset(path: str | Path) -> List[Dict[str, str]]:
    """
    Load test/private test dataset from CSV file.
    
    Format:
        id,text
        1,"Sample text for testing"
        2,"Another sample"
    
    Args:
        path: Path to CSV file
        
    Returns:
        List of dicts with 'id' and 'text' keys
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    
    try:
        df = pd.read_csv(path)
        
        if "id" not in df.columns or "text" not in df.columns:
            raise ValueError("Test dataset must have 'id' and 'text' columns")
        
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "id": str(row["id"]),
                "text": str(row["text"])
            })
        
        logger.info(f"Loaded {len(samples)} test samples from {path}")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading test dataset from {path}: {e}")
        raise


def save_predictions(
    predictions: List[DocumentPrediction],
    output_path: str | Path,
    format: str = "json"
) -> None:
    """
    Save predictions to file.
    
    Args:
        predictions: List of DocumentPrediction objects
        output_path: Path to output file
        format: 'json' or 'csv'
        
    Raises:
        ValueError: If format is invalid
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        data = [
            {
                "id": pred.doc_id,
                "text": pred.text,
                "predictions": pred.to_span_tuples()
            }
            for pred in predictions
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(predictions)} predictions to {output_path} (JSON)")
        
    elif format == "csv":
        rows = []
        for pred in predictions:
            span_tuples = pred.to_span_tuples()
            rows.append({
                "id": pred.doc_id,
                "text": pred.text,
                "predictions": json.dumps(span_tuples)
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {output_path} (CSV)")
        
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")


def load_predictions(path: str | Path, format: str = "json") -> List[DocumentPrediction]:
    """
    Load predictions from file.
    
    Args:
        path: Path to predictions file
        format: 'json' or 'csv'
        
    Returns:
        List of DocumentPrediction objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    
    predictions = []
    
    if format == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            entities = [
                EntityModel(start=s, end=e, category=c)
                for s, e, c in item["predictions"]
            ]
            pred = DocumentPrediction(
                doc_id=item["id"],
                text=item["text"],
                predictions=entities
            )
            predictions.append(pred)
    
    elif format == "csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            span_tuples = json.loads(row["predictions"])
            entities = [
                EntityModel(start=s, end=e, category=c)
                for s, e, c in span_tuples
            ]
            pred = DocumentPrediction(
                doc_id=row["id"],
                text=row["text"],
                predictions=entities
            )
            predictions.append(pred)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
    
    logger.info(f"Loaded {len(predictions)} predictions from {path}")
    return predictions


def parse_entities_from_string(entities_str: str) -> List[Span]:
    """
    Parse entities from JSON string representation.
    
    Expected format:
        '[{"start": 0, "end": 5, "category": "LABEL"}]'
    
    Args:
        entities_str: JSON string
        
    Returns:
        List of Span objects
        
    Raises:
        ValueError: If parsing fails
    """
    if not entities_str or not entities_str.strip():
        return []
    
    try:
        entities_list = json.loads(entities_str)
        spans = []
        for entity_dict in entities_list:
            entity = EntityModel(**entity_dict)
            span = entity.to_span()
            spans.append(span)
        return spans
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse entities string: {e}")


def validate_spans(text: str, spans: List[SpanTuple] | List[Span]) -> bool:
    """
    Validate that spans are valid for the given text.
    
    Checks:
        - 0 <= start < end <= len(text)
        - category is not empty
        - spans are sorted by start position
        - no overlapping spans
        - start and end point to valid text positions
    
    Args:
        text: Input text
        spans: List of span tuples (start, end, category) or Span objects
        
    Returns:
        True if all spans are valid
        
    Raises:
        ValueError: If any span is invalid
    """
    text_len = len(text)
    
    # Convert Span objects to tuples if needed
    if spans and isinstance(spans[0], Span):
        span_tuples = [s.to_tuple() for s in spans]
    else:
        span_tuples = spans
    
    if not span_tuples:
        return True
    
    # Check each span
    for start, end, category in span_tuples:
        # Check boundaries
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Span positions must be integers: ({start}, {end})")
        
        if start < 0:
            raise ValueError(f"Span start cannot be negative: {start}")
        
        if end <= start:
            raise ValueError(f"Span end must be greater than start: ({start}, {end})")
        
        if end > text_len:
            raise ValueError(f"Span end exceeds text length: {end} > {text_len}")
        
        # Check category
        if not category or not isinstance(category, str):
            raise ValueError(f"Category must be non-empty string: {category!r}")
        
        # Check text is not empty at span
        span_text = text[start:end]
        if not span_text or not span_text.strip():
            raise ValueError(f"Span [{start}:{end}] contains only whitespace: {span_text!r}")
    
    # Check ordering
    prev_start = -1
    for start, end, _ in span_tuples:
        if start < prev_start:
            raise ValueError(f"Spans not sorted by start position")
        prev_start = start
    
    # Check overlaps
    for i in range(len(span_tuples)):
        s1_start, s1_end, _ = span_tuples[i]
        for j in range(i + 1, len(span_tuples)):
            s2_start, s2_end, _ = span_tuples[j]
            if s1_end > s2_start:
                raise ValueError(f"Overlapping spans: ({s1_start}, {s1_end}) and ({s2_start}, {s2_end})")
    
    return True


def ensure_sorted_non_overlapping(spans: List[SpanTuple]) -> List[SpanTuple]:
    """
    Sort spans by start position and ensure no overlaps (remove contained spans).
    
    Args:
        spans: List of span tuples
        
    Returns:
        Sorted and deduplicated span list
    """
    if not spans:
        return []
    
    # Sort by start, then by end (descending)
    sorted_spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    
    # Remove overlaps and duplicates
    result = []
    for start, end, category in sorted_spans:
        # Check if this span is already covered by a previous span
        skip = False
        for r_start, r_end, r_cat in result:
            if r_start == start and r_end == end and r_cat == category:
                # Exact duplicate
                skip = True
                break
            if r_start <= start and end <= r_end:
                # Already covered by larger span
                skip = True
                break
        
        if not skip:
            # Remove any spans that this new span covers
            result = [
                (rs, re, rc) for rs, re, rc in result
                if not (start <= rs and re <= end and (rs != start or re != end))
            ]
            result.append((start, end, category))
    
    return result
