# File: ml/rules/__init__.py
import re
from typing import List, Tuple

def apply_rules(text: str) -> List[Tuple[int, int, str]]:
    """Apply regex-based rules for PII detection.
    
    Args:
        text: Input text
    
    Returns:
        List of (start_char, end_char, category) tuples
    """
    if not text:
        return []
    
    entities = []
    
    # Russian passport: series (4 digits) + number (6 digits)
    # Formats: "4510 654321", "4510654321", "серия 4510 номер 654321"
    passport_pattern = r'\b\d{4}\s?\d{6}\b'
    for match in re.finditer(passport_pattern, text):
        entities.append((match.start(), match.end(), "Паспортные данные РФ"))
    
    # СНИЛС: XXX-XXX-XXX YY format
    # Formats: "123-456-789 12", "123-456-78912", "12345678912"
    snils_pattern = r'\b\d{3}-\d{3}-\d{3}\s?\d{2}\b'
    for match in re.finditer(snils_pattern, text):
        entities.append((match.start(), match.end(), "СНИЛС"))
    
    # Alternative SNILS format (no dashes)
    snils_nodash_pattern = r'\b\d{11}\b'
    for match in re.finditer(snils_nodash_pattern, text):
        # Avoid matching INN (12 digits) or phone numbers
        if not re.match(r'\d{12}', text[match.start():match.end()+1]):
            entities.append((match.start(), match.end(), "СНИЛС"))
    
    # ИНН: 12 digits for individual, 10 for organization
    inn_pattern = r'\b\d{12}\b|\b\d{10}\b'
    for match in re.finditer(inn_pattern, text):
        entities.append((match.start(), match.end(), "ИНН"))
    
    # Phone numbers (Russian formats)
    # +7XXXXXXXXXX, 8XXXXXXXXXX, +7 (XXX) XXX-XX-XX, etc.
    phone_pattern = r'(?:\+7|8|7)[\s\-]?\(?(\d{3})\)?[\s\-]?(\d{3})[\s\-]?(\d{2})[\s\-]?(\d{2})'
    for match in re.finditer(phone_pattern, text):
        entities.append((match.start(), match.end(), "Номер телефона"))
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append((match.start(), match.end(), "Email"))
    
    # Date of birth (DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY)
    date_pattern = r'\b\d{2}[\./\-]\d{2}[\./\-]\d{4}\b'
    for match in re.finditer(date_pattern, text):
        entities.append((match.start(), match.end(), "Дата рождения"))
    
    # Bank card numbers (16 digits, may have spaces or dashes every 4 digits)
    card_pattern = r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
    for match in re.finditer(card_pattern, text):
        entities.append((match.start(), match.end(), "Банковские данные"))
    
    # Bank account numbers (20 digits)
    account_pattern = r'\b\d{20}\b'
    for match in re.finditer(account_pattern, text):
        entities.append((match.start(), match.end(), "Банковские данные"))
    
    # Deduplicate overlapping matches (keep first match)
    entities = _deduplicate_spans(entities)
    
    return entities

def _deduplicate_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove duplicate or overlapping spans, keeping the first occurrence."""
    if not spans:
        return []
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    deduplicated = []
    for span in sorted_spans:
        start, end, category = span
        
        # Check if this span overlaps with any already added
        has_overlap = False
        for existing_start, existing_end, _ in deduplicated:
            # Check for overlap
            if not (end <= existing_start or start >= existing_end):
                has_overlap = True
                break
        
        if not has_overlap:
            deduplicated.append(span)
    
    return deduplicated
