"""Data readers for loading raw datasets."""

import ast
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logging_utils import get_logger
from src.data.schemas import Entity, Document

logger = get_logger(__name__)


def parse_entities_string(entities_str: str) -> List[Entity]:
    """
    Parse entities from string representation.

    Supported formats:
    - Python literal list of tuples:
      [(26, 69, 'API ключи')]
    - Python literal list of lists:
      [[26, 69, 'API ключи']]
    - JSON list of dicts:
      [{"start": 26, "end": 69, "category": "API ключи"}]
    - Empty list:
      []

    Args:
        entities_str: String representation of entity annotations

    Returns:
        List of Entity objects
    """
    if entities_str is None:
        return []

    entities_str = str(entities_str).strip()

    if not entities_str or entities_str.lower() == "nan" or entities_str == "[]":
        return []

    def _convert_parsed_entities(parsed) -> List[Entity]:
        if not isinstance(parsed, list):
            logger.warning(f"Expected list, got {type(parsed).__name__}")
            return []

        entities: List[Entity] = []

        for item in parsed:
            try:
                if isinstance(item, dict):
                    entity = Entity(**item)
                    entities.append(entity)
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    start, end, category = item
                    entity = Entity(
                        start=int(start),
                        end=int(end),
                        category=str(category)
                    )
                    entities.append(entity)
                else:
                    logger.warning(f"Unsupported entity format: {item}")
            except Exception as e:
                logger.warning(f"Failed to parse entity {item}: {e}")

        return entities

    try:
        parsed = json.loads(entities_str)
        entities = _convert_parsed_entities(parsed)
        if entities or parsed == []:
            return entities
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(entities_str)
        return _convert_parsed_entities(parsed)
    except Exception as e:
        logger.warning(f"Failed to parse entities string: {e}; value={entities_str[:200]}")
        return []


def load_train_dataset(
    path: str | Path,
    validate: bool = True
) -> List[Document]:
    """
    Load training dataset from TSV file.

    Expected columns: text, target (or entities)

    Supported target formats:
    - [(start, end, category), ...]
    - [{"start": ..., "end": ..., "category": ...}, ...]

    Args:
        path: Path to TSV file
        validate: Whether to validate entity span ranges

    Returns:
        List of Document objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Train dataset not found: {path}")

    logger.info(f"Loading train dataset from {path}")

    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

        if "target" in df.columns:
            entities_col = "target"
        elif "entities" in df.columns:
            entities_col = "entities"
        else:
            raise ValueError(
                f"Missing 'target' or 'entities' column. Columns: {df.columns.tolist()}"
            )

        if "text" not in df.columns:
            raise ValueError(f"Missing 'text' column. Columns: {df.columns.tolist()}")

        documents: List[Document] = []
        errors = 0

        for idx, row in df.iterrows():
            try:
                text = str(row["text"]).strip()
                if not text:
                    logger.warning(f"Row {idx}: empty text, skipping")
                    errors += 1
                    continue

                raw_entities = row[entities_col]
                entities = parse_entities_string(raw_entities)

                if validate:
                    valid_entities: List[Entity] = []
                    for entity in entities:
                        if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
                            logger.warning(
                                f"Row {idx}: entity span ({entity.start}, {entity.end}) "
                                f"out of bounds for text length {len(text)}"
                            )
                            errors += 1
                            continue
                        valid_entities.append(entity)
                    entities = valid_entities

                doc_id = str(row["id"]) if "id" in df.columns else str(idx)

                doc = Document(
                    text=text,
                    entities=entities,
                    doc_id=doc_id
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

    Args:
        path: Path to CSV file

    Returns:
        List of Document objects without entities
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    logger.info(f"Loading test dataset from {path}")

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        if "text" not in df.columns:
            raise ValueError(f"Missing 'text' column. Columns: {df.columns.tolist()}")

        if "id" not in df.columns:
            raise ValueError(f"Missing 'id' column. Columns: {df.columns.tolist()}")

        documents: List[Document] = []
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

    documents: List[Document] = []

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
