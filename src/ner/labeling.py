"""BIO labeling utilities for NER."""

from typing import List, Tuple, Dict
import json

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BIOLabeler:
    """Convert between span and BIO representations."""
    
    def __init__(self, categories: List[str]):
        """
        Initialize labeler.
        
        Args:
            categories: List of entity categories (e.g., ["PASSPORT", "PHONE", ...])
        """
        self.categories = sorted(categories)
        self.tags = self._create_bio_tags()
        self.tag_to_id = {tag: idx for idx, tag in enumerate(self.tags)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
    
    def _create_bio_tags(self) -> List[str]:
        """Create BIO tag list."""
        tags = ["O"]
        for category in self.categories:
            tags.append(f"B-{category}")
            tags.append(f"I-{category}")
        return tags
    
    @property
    def num_labels(self) -> int:
        """Get number of label types."""
        return len(self.tags)
    
    def encode_tag(self, tag: str) -> int:
        """Encode tag to ID."""
        return self.tag_to_id.get(tag, 0)  # 0 = "O" for unknown
    
    def decode_tag(self, tag_id: int) -> str:
        """Decode ID to tag."""
        return self.id_to_tag.get(tag_id, "O")
    
    def get_tags(self) -> List[str]:
        """Get all tags."""
        return self.tags.copy()
    
    def save(self, path: str) -> None:
        """Save labeler to JSON."""
        data = {
            "categories": self.categories,
            "tags": self.tags,
            "tag_to_id": self.tag_to_id,
            "id_to_tag": {str(k): v for k, v in self.id_to_tag.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved labeler to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BIOLabeler":
        """Load labeler from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labeler = cls(data["categories"])
        return labeler


def align_labels_to_tokens(
    token_offsets: List[Tuple[int, int]],
    labels: List[int],
    special_token_mask: List[int] = None
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Filter out special tokens from labels and offsets.
    
    Args:
        token_offsets: (start, end) offset for each token
        labels: Label ID for each token
        special_token_mask: 1 for special, 0 for real token (optional)
        
    Returns:
        (filtered_labels, filtered_offsets)
    """
    if special_token_mask is not None:
        filtered = [
            (label, offset)
            for label, offset, is_special in zip(labels, token_offsets, special_token_mask)
            if is_special == 0
        ]
    else:
        # Filter by zero-length offsets
        filtered = [
            (label, offset)
            for label, offset in zip(labels, token_offsets)
            if offset[0] != offset[1]
        ]
    
    if not filtered:
        return [], []
    
    labels_out, offsets_out = zip(*filtered)
    return list(labels_out), list(offsets_out)


def merge_tokens_to_spans(
    token_labels: List[int],
    token_offsets: List[Tuple[int, int]],
    id_to_tag: Dict[int, str],
    text: str = None
) -> List[Tuple[int, int, str]]:
    """
    Convert token-level BIO labels to character-level spans.
    
    Args:
        token_labels: BIO label IDs for each token
        token_offsets: Character offsets for each token
        id_to_tag: Mapping from label ID to tag string
        text: Original text (optional, for validation)
        
    Returns:
        List of (start, end, category) tuples
    """
    if not token_labels or not token_offsets:
        return []
    
    spans = []
    current_entity_start = None
    current_entity_category = None
    prev_token_end = None
    
    for token_idx, (label_id, (token_start, token_end)) in enumerate(
        zip(token_labels, token_offsets)
    ):
        tag = id_to_tag.get(label_id, "O")
        
        if tag == "O":
            # Outside tag - close any open entity
            if current_entity_start is not None:
                spans.append((
                    current_entity_start,
                    prev_token_end,
                    current_entity_category
                ))
                current_entity_start = None
                current_entity_category = None
        
        elif tag.startswith("B-"):
            # Beginning tag
            # Close any open entity first
            if current_entity_start is not None:
                spans.append((
                    current_entity_start,
                    prev_token_end,
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
                # I- without preceding B- (treat as B-)
                current_entity_start = token_start
                current_entity_category = category
            elif current_entity_category != category:
                # Category mismatch - close and start new
                spans.append((
                    current_entity_start,
                    prev_token_end,
                    current_entity_category
                ))
                current_entity_start = token_start
                current_entity_category = category
        
        prev_token_end = token_end
    
    # Close remaining open entity
    if current_entity_start is not None:
        spans.append((
            current_entity_start,
            prev_token_end,
            current_entity_category
        ))
    
    # Sort by start
    spans.sort(key=lambda x: x[0])
    
    # Remove duplicates and validate
    spans = _deduplicate_spans(spans)
    
    if text:
        spans = _validate_spans(spans, text)
    
    return spans


def _deduplicate_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove duplicate and overlapping spans."""
    if not spans:
        return []
    
    # Sort by (start, -length) to keep longer spans
    sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    
    result = []
    covered = set()
    
    for start, end, category in sorted_spans:
        # Check if covered by existing span
        is_covered = any(pos in covered for pos in range(start, end))
        
        if not is_covered:
            result.append((start, end, category))
            for pos in range(start, end):
                covered.add(pos)
    
    return result


def _validate_spans(
    spans: List[Tuple[int, int, str]],
    text: str
) -> List[Tuple[int, int, str]]:
    """Validate spans are within text bounds."""
    text_len = len(text)
    valid = []
    
    for start, end, category in spans:
        if 0 <= start < end <= text_len:
            span_text = text[start:end]
            # Check not all whitespace
            if span_text.strip():
                valid.append((start, end, category))
    
    return valid
