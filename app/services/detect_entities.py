# File: app/services/detect_entities.py
from typing import List, Tuple, Dict
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.pipelines.infer_ner import NERModel
from ml.rules import apply_rules

# Global model instance (load once)
_ner_model = None

def get_ner_model() -> NERModel:
    """Get or initialize NER model singleton."""
    global _ner_model
    if _ner_model is None:
        _ner_model = NERModel()
    return _ner_model

def detect_entities(text: str) -> List[Tuple[int, int, str]]:
    """Detect PII entities in text using NER model + rules.
    
    Args:
        text: Input text to analyze
    
    Returns:
        List of (start_char, end_char, category) tuples, sorted by start position
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Get NER predictions
    ner_model = get_ner_model()
    ner_entities = ner_model.predict(text)
    
    # Get rule-based predictions
    rule_entities = apply_rules(text)
    
    # Merge and deduplicate
    merged = _merge_entities(ner_entities, rule_entities)
    
    return sorted(merged, key=lambda x: x[0])

def _merge_entities(
    ner_entities: List[Tuple[int, int, str]],
    rule_entities: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Merge NER and rule-based entities, preferring NER on overlap.
    
    Strategy:
    - If spans don't overlap, keep both
    - If spans overlap, prefer NER result
    """
    if not rule_entities:
        return ner_entities
    
    if not ner_entities:
        return rule_entities
    
    # Convert NER entities to set for quick lookup
    merged = list(ner_entities)
    ner_spans = [(start, end) for start, end, _ in ner_entities]
    
    # Add rule entities that don't overlap with NER
    for rule_entity in rule_entities:
        rule_start, rule_end, rule_cat = rule_entity
        
        # Check for overlap with any NER entity
        has_overlap = False
        for ner_start, ner_end in ner_spans:
            # Check if ranges overlap
            if not (rule_end <= ner_start or rule_start >= ner_end):
                has_overlap = True
                break
        
        # Add if no overlap
        if not has_overlap:
            merged.append(rule_entity)
    
    # Deduplicate exact matches
    return list(set(merged))

def mask_text(text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """Replace PII entities with placeholders.
    
    Args:
        text: Original text
        entities: List of (start_char, end_char, category) tuples
    
    Returns:
        Tuple of (masked_text, mapping) where:
        - masked_text: Text with PII replaced by [PII_0], [PII_1], etc.
        - mapping: Dict mapping placeholder -> (original_value, category)
    """
    if not entities:
        return text, {}
    
    # Sort entities by start position (reverse for replacement)
    sorted_entities = sorted(entities, key=lambda x: x[0], reverse=True)
    
    masked_text = text
    mapping = {}
    
    for idx, (start, end, category) in enumerate(reversed(sorted_entities)):
        original_value = text[start:end]
        placeholder = f"[PII_{idx}]"
        
        # Replace in text (working backwards to preserve indices)
        masked_text = masked_text[:start] + placeholder + masked_text[end:]
        
        # Store mapping
        mapping[placeholder] = (original_value, category)
    
    return masked_text, mapping

def demask_text(masked_text: str, mapping: Dict[str, Tuple[str, str]]) -> str:
    """Restore original PII values from masked text.
    
    Args:
        masked_text: Text with [PII_N] placeholders
        mapping: Dict mapping placeholder -> (original_value, category)
    
    Returns:
        Text with placeholders replaced by original values
    """
    if not mapping:
        return masked_text
    
    result = masked_text
    
    # Replace all placeholders
    for placeholder, (original_value, _) in mapping.items():
        result = result.replace(placeholder, original_value)
    
    return result

def demask_streaming(masked_tokens: List[str], mapping: Dict[str, Tuple[str, str]]) -> List[str]:
    """Demask tokens for streaming output.
    
    Handles partial placeholders that may appear across token boundaries.
    
    Args:
        masked_tokens: List of token strings (may contain [PII_N] placeholders)
        mapping: Dict mapping placeholder -> (original_value, category)
    
    Returns:
        List of tokens with placeholders replaced
    """
    if not mapping:
        return masked_tokens
    
    # Join tokens to handle split placeholders
    text = ''.join(masked_tokens)
    
    # Replace placeholders
    for placeholder, (original_value, _) in mapping.items():
        text = text.replace(placeholder, original_value)
    
    # For streaming, we could split back, but typically return full text
    return [text]

def batch_detect_entities(texts: List[str]) -> List[List[Tuple[int, int, str]]]:
    """Detect entities in multiple texts efficiently.
    
    Args:
        texts: List of input texts
    
    Returns:
        List of entity lists, one per input text
    """
    if not texts:
        return []
    
    # Batch NER prediction
    ner_model = get_ner_model()
    batch_ner_entities = ner_model.predict_batch(texts)
    
    # Apply rules to each text
    results = []
    for text, ner_entities in zip(texts, batch_ner_entities):
        rule_entities = apply_rules(text)
        merged = _merge_entities(ner_entities, rule_entities)
        results.append(sorted(merged, key=lambda x: x[0]))
    
    return results
