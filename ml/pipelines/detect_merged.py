"""
Unified detection interface for alpha-pii-guard.

Provides detect_regex, detect_ner, and detect_merged functions with consistent output format.
"""

from typing import List, Tuple
import re

from ml.merge.resolver import merge_entities
from ml.merge.label_map import normalize_label


def detect_regex(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect PII using regex patterns.
    
    Returns char-level spans with normalized labels.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of (start, end, label) tuples
    """
    spans = []
    
    # Email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        spans.append((match.start(), match.end(), "Email"))
    
    # Russian phone numbers
    phone_patterns = [
        r'\+7\s?\(?\d{3}\)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}',
        r'8\s?\(?\d{3}\)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}',
        r'\b\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
    ]
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), "Номер телефона"))
    
    # Russian passport: 4 digits space 6 digits
    passport_pattern = r'\b\d{4}\s?\d{6}\b'
    for match in re.finditer(passport_pattern, text):
        spans.append((match.start(), match.end(), "Паспортные данные"))
    
    # INN: 10 or 12 digits
    inn_pattern = r'\b\d{10}(?:\d{2})?\b'
    for match in re.finditer(inn_pattern, text):
        # Context check to reduce false positives
        start_ctx = max(0, match.start() - 20)
        end_ctx = min(len(text), match.end() + 20)
        context = text[start_ctx:end_ctx].lower()
        if 'инн' in context or 'ИНН' in text[start_ctx:end_ctx]:
            spans.append((match.start(), match.end(), "Сведения об ИНН"))
    
    # SNILS: 11 digits with dashes XXX-XXX-XXX XX
    snils_pattern = r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b'
    for match in re.finditer(snils_pattern, text):
        start_ctx = max(0, match.start() - 20)
        end_ctx = min(len(text), match.end() + 20)
        context = text[start_ctx:end_ctx].lower()
        if 'снилс' in context or 'СНИЛС' in text[start_ctx:end_ctx]:
            spans.append((match.start(), match.end(), "СНИЛС клиента"))
    
    # Card number: 13-19 digits with optional spaces/dashes
    card_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?:[\s-]?\d{3})?\b'
    for match in re.finditer(card_pattern, text):
        spans.append((match.start(), match.end(), "Номер карты"))
    
    # Bank account: 20 digits
    account_pattern = r'\b\d{20}\b'
    for match in re.finditer(account_pattern, text):
        start_ctx = max(0, match.start() - 30)
        end_ctx = min(len(text), match.end() + 30)
        context = text[start_ctx:end_ctx].lower()
        if 'счет' in context or 'счёт' in context or 'р/с' in context or 'рс' in context:
            spans.append((match.start(), match.end(), "Номер банковского счета"))
    
    # CVV/CVC: 3-4 digits with context
    cvv_pattern = r'\b\d{3,4}\b'
    for match in re.finditer(cvv_pattern, text):
        start_ctx = max(0, match.start() - 15)
        end_ctx = min(len(text), match.end() + 15)
        context = text[start_ctx:end_ctx].lower()
        if 'cvv' in context or 'cvc' in context or 'цвв' in context:
            spans.append((match.start(), match.end(), "CVV/CVC"))
    
    # PIN code: 4-6 digits with context
    pin_pattern = r'\b\d{4,6}\b'
    for match in re.finditer(pin_pattern, text):
        start_ctx = max(0, match.start() - 15)
        end_ctx = min(len(text), match.end() + 15)
        context = text[start_ctx:end_ctx].lower()
        if 'pin' in context or 'пин' in context:
            spans.append((match.start(), match.end(), "ПИН код"))
    
    # Card expiry: MM/YY or MM/YYYY
    expiry_pattern = r'\b(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})\b'
    for match in re.finditer(expiry_pattern, text):
        start_ctx = max(0, match.start() - 25)
        end_ctx = min(len(text), match.end() + 25)
        context = text[start_ctx:end_ctx].lower()
        if any(kw in context for kw in ['срок', 'действ', 'expir', 'valid', 'карт']):
            spans.append((match.start(), match.end(), "Дата окончания срока действия карты"))
    
    # One-time codes: 4-6 digits with context
    otp_pattern = r'\b\d{4,6}\b'
    for match in re.finditer(otp_pattern, text):
        start_ctx = max(0, match.start() - 20)
        end_ctx = min(len(text), match.end() + 20)
        context = text[start_ctx:end_ctx].lower()
        if any(kw in context for kw in ['код', 'sms', 'смс', 'otp', 'подтверждени']):
            spans.append((match.start(), match.end(), "Одноразовые коды"))
    
    # API keys: hex or base64 patterns with length constraints
    api_hex_pattern = r'\b[A-Fa-f0-9]{32,64}\b'
    api_base64_pattern = r'\b[A-Za-z0-9+/]{40,}={0,2}\b'
    for match in re.finditer(api_hex_pattern, text):
        start_ctx = max(0, match.start() - 20)
        end_ctx = min(len(text), match.end() + 20)
        context = text[start_ctx:end_ctx].lower()
        if 'api' in context or 'key' in context or 'token' in context or 'ключ' in context:
            spans.append((match.start(), match.end(), "API ключи"))
    
    for match in re.finditer(api_base64_pattern, text):
        start_ctx = max(0, match.start() - 20)
        end_ctx = min(len(text), match.end() + 20)
        context = text[start_ctx:end_ctx].lower()
        if 'api' in context or 'key' in context or 'token' in context or 'ключ' in context:
            spans.append((match.start(), match.end(), "API ключи"))
    
    # Driver's license (Russian): 4 digits space 6 digits
    driver_pattern = r'\b\d{2}\s?\d{2}\s?\d{6}\b'
    for match in re.finditer(driver_pattern, text):
        start_ctx = max(0, match.start() - 30)
        end_ctx = min(len(text), match.end() + 30)
        context = text[start_ctx:end_ctx].lower()
        if 'водител' in context or 'ву' in context or 'прав' in context:
            spans.append((match.start(), match.end(), "Водительское удостоверение"))
    
    # Birth certificate: Roman numerals + digits
    birth_cert_pattern = r'\b[IVXLCDM]+-[А-Я]{2}\s?\d{6}\b'
    for match in re.finditer(birth_cert_pattern, text):
        spans.append((match.start(), match.end(), "Свидетельство о рождении"))
    
    # Residence permit: series and number
    residence_pattern = r'\b\d{2}\s?\d{2}\s?\d{7}\b'
    for match in re.finditer(residence_pattern, text):
        start_ctx = max(0, match.start() - 30)
        end_ctx = min(len(text), match.end() + 30)
        context = text[start_ctx:end_ctx].lower()
        if 'вид на жительство' in context or 'внж' in context:
            spans.append((match.start(), match.end(), "Серия и номер вида на жительство"))
    
    # Temporary ID
    temp_id_pattern = r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'
    for match in re.finditer(temp_id_pattern, text):
        start_ctx = max(0, match.start() - 40)
        end_ctx = min(len(text), match.end() + 40)
        context = text[start_ctx:end_ctx].lower()
        if 'временное удостоверение' in context or 'временн' in context:
            spans.append((match.start(), match.end(), "Временное удостоверение личности"))
    
    # Magnetic stripe data: track1/track2 format
    magstripe_pattern = r'%[A-Z]?\d{13,19}\^[^\^]+\^[^\?]+\?'
    for match in re.finditer(magstripe_pattern, text):
        spans.append((match.start(), match.end(), "Содержимое магнитной полосы"))
    
    return spans


def detect_ner(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect PII using NER model.
    
    Returns char-level spans with normalized labels.
    
    NOTE: This is an adapter stub. Real implementation should:
    1. Load NER model
    2. Run inference
    3. Decode token-level BIO predictions to char-level spans
    4. Normalize labels using normalize_label()
    5. Validate and trim boundaries
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of (start, end, label) tuples
    """
    # TODO: Replace with actual NER inference
    # from ml.pipelines.infer_ner import run_inference
    # 
    # token_predictions = run_inference(text, model)
    # char_spans = decode_bio_to_char_spans(token_predictions, text)
    # normalized = [(s, e, normalize_label(label)) for s, e, label in char_spans]
    # return validate_and_trim_spans(normalized, text)
    
    # Stub implementation
    return []


def detect_merged(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect PII using merged regex + NER approach.
    
    Returns char-level spans with normalized labels.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of (start, end, label) tuples
    """
    regex_spans = detect_regex(text)
    ner_spans = detect_ner(text)
    merged = merge_entities(regex_spans, ner_spans, text)
    return merged
