"""Data readers for loading raw datasets."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from src.utils.logging_utils import get_logger
from src.data.schemas import Entity, Document

logger = get_logger(__name__)


def parse_entities_string(entities_str: str) -> List[Entity]:
    """
    Parse entities from string representation.
    
    Supports formats:
    - JSON: '[{"start": 0, "end": 5, "category": "LABEL"}]'
    - Safe fallback if malformed
    
    Args:
        entities_str: String representation of entities list
        
    Returns:
        List of Entity objects
    """
    if not entities_str or not str(entities_str).strip():
        return []
    
    entities_str = str(entities_str).strip()
    
    try:
        entities_list = json.loads(entities_str)
        if not isinstance(entities_list, list):
            logger.warning(f"Expected list, got {type(entities_list).__name__}")
            return []
        
        entities = []
        for entity_dict in entities_list:
            try:
                entity = Entity(**entity_dict)
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity dict {entity_dict}: {e}")
                continue
        
        return entities
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse entities JSON: {e}")
        return []


def load_train_dataset(
    path: str | Path,
    validate: bool = True
) -> List[Document]:
    """
    Load training dataset from TSV file.
    
    Expected columns: text, target (or entities)
    
    Format:
        text    target
        "Sample text"    [{"start": 0, "end": 5, "category": "LABEL"}]
    
    Args:
        path: Path to TSV file
        validate: Whether to validate entities span ranges
        
    Returns:
        List of Document objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Train dataset not found: {path}")
    
    logger.info(f"Loading train dataset from {path}")
    
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, quotechar='"')
        
        # Handle column naming flexibility
        if "target" in df.columns:
            entities_col = "target"
        elif "entities" in df.columns:
            entities_col = "entities"
        else:
            raise ValueError(f"Missing 'target' or 'entities' column. Columns: {df.columns.tolist()}")
        
        if "text" not in df.columns:
            raise ValueError(f"Missing 'text' column. Columns: {df.columns.tolist()}")
        
        documents = []
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                text = str(row["text"]).strip()
                if not text:
                    logger.warning(f"Row {idx}: empty text, skipping")
                    errors += 1
                    continue
                
                entities = parse_entities_string(row[entities_col])
                
                # Validate entities against text
                if validate:
                    for entity in entities:
                        if entity.start < 0 or entity.end > len(text):
                            logger.warning(
                                f"Row {idx}: entity span ({entity.start}, {entity.end}) "
                                f"out of bounds for text length {len(text)}"
                            )
                            errors += 1
                            continue
                
                doc_id = row.get("id", str(idx)) if "id" in df.columns else str(idx)
                
                doc = Document(
                    text=text,
                    entities=entities,
                    doc_id=str(doc_id)
                )
                documents.append(doc)
            
            except Exception as e:
                logger.warning(f"Row {idx}: failed to parse: {e}")
                errors += 1
                continue
        
        logger.info(
            f"Loaded {len(documents)} documents (skipped {errors} due to errors)"
        )
        return documents
    
    except Exception as e:
        logger.error(f"Error loading train dataset: {e}")
        raise


def load_test_dataset(
    path: str | Path
) -> List[Document]:
    """
    Load test/private test dataset from CSV file.
    
    Expected columns: id, text
    
    Format:
        id,text
        1,"Sample text for testing"
        2,"Another sample"
    
    Args:
        path: Path to CSV file
        
    Returns:
        List of Document objects (without entities)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    
    logger.info(f"Loading test dataset from {path}")
    
    try:
        df = pd.read_csv(path, dtype=str)
        
        if "text" not in df.columns:
            raise ValueError(f"Missing 'text' column. Columns: {df.columns.tolist()}")
        
        if "id" not in df.columns:
            raise ValueError(f"Missing 'id' column. Columns: {df.columns.tolist()}")
        
        documents = []
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                text = str(row["text"]).strip()
                if not text:
                    logger.warning(f"Row {idx}: empty text, skipping")
                    errors += 1
                    continue
                
                doc_id = str(row["id"])
                
                doc = Document(
                    text=text,
                    entities=[],
                    doc_id=doc_id
                )
                documents.append(doc)
            
            except Exception as e:
                logger.warning(f"Row {idx}: failed to parse: {e}")
                errors += 1
                continue
        
        logger.info(
            f"Loaded {len(documents)} test documents (skipped {errors} due to errors)"
        )
        return documents
    
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        raise


def load_jsonl_documents(path: str | Path) -> List[Document]:
    """
    Load documents from JSONL format.
    
    Each line: {"text": "...", "entities": [...], "doc_id": "..."}
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of Document objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    
    documents = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                doc = Document(**data)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Line {line_no}: failed to parse: {e}")
                continue
    
    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents


def save_jsonl_documents(documents: List[Document], path: str | Path) -> None:
    """
    Save documents to JSONL format.
    
    Args:
        documents: List of Document objects
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            line = doc.model_dump_json(ensure_ascii=False)
            f.write(line + "\n")
    
    logger.info(f"Saved {len(documents)} documents to {path}")
