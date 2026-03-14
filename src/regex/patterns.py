"""Compiled regex patterns for PII detection."""

import re
from typing import Dict, List, Tuple, Pattern

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RegexPatterns:
    """Collection of regex patterns for PII detection."""
    
    # ==================== API & AUTH ====================
    
    # API keys (common formats)
    API_KEY_PATTERNS = [
        # AWS-like keys (AKIA + 16 chars alphanumeric)
        r'AKIA[0-9A-Z]{16}',
        # Generic API key format (api_key= or apikey= followed by hex/alphanumeric)
        r'(?:api[_-]?key|apikey)\s*[:=]\s*[a-zA-Z0-9_\-]{20,}',
        # SK_LIVE or SK_TEST patterns (Stripe-like)
        r'(?:sk_live|sk_test|rk_live|rk_test)_[a-zA-Z0-9_]{20,}',
    ]
    
    # CVV/CVC codes (3-4 digits, often preceded by specific keywords)
    CVV_CVC_PATTERNS = [
        # Explicit CVV/CVC followed by digits
        r'(?:cvv|cvc|security code)\s*[:=]?\s*\d{3,4}',
    ]
    
    # PIN codes (often preceded by keywords, 4-6 digits)
    PIN_PATTERNS = [
        r'(?:pin|пин)\s*[:=]?\s*\d{4,6}',
    ]
    
    # Passwords (context-based, very conservative)
    PASSWORD_PATTERNS = [
        r'(?:password|пароль)\s*[:=]\s*[^\s]{8,}',
    ]
    
    # ==================== COMMUNICATIONS ====================
    
    # Email addresses (RFC 5322 simplified)
    EMAIL_PATTERNS = [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    ]
    
    # Phone numbers (Russian and international formats)
    PHONE_PATTERNS = [
        # Russian: +7 999 123 45 67 or variations
        r'\+7[\s-]?(?:9|8)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}',
        # Russian: 8 999 123 45 67 or variations
        r'8[\s-]?(?:9|8)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}',
        # International: +1-555-123-4567 or similar
        r'\+[1-9]\d{0,2}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',
        # Generic: (XXX) XXX-XXXX
        r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',
    ]
    
    # ==================== FINANCIAL ====================
    
    # Bank card numbers (16-19 digits, with optional spaces/dashes)
    CARD_PATTERNS = [
        r'\b(?:\d[\s-]?){15,18}\d\b',  # 16-19 digit card number
        # More specific: common prefixes (Visa 4, Mastercard 51-55, Amex 34/37)
        r'\b(?:4[0-9]|5[1-5]|3[47])[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    ]
    
    # Bank account/settlement account numbers (20-digit Russian format)
    ACCOUNT_PATTERNS = [
        r'\b\d{20}\b',  # Generic 20-digit account
        # Russian settlement account (40817... format)
        r'\b40[0-9]{2}[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    ]
    
    # Card expiration dates (MM/YY or MM-YY)
    CARD_EXPIRY_PATTERNS = [
        r'(?:exp|expiry|expires?)[\s:=]*(?:0[1-9]|1[0-2])/\d{2}',
        r'(?:0[1-9]|1[0-2])/\d{2}(?:\s+or\s+)?(?:expires)?',
    ]
    
    # ==================== RUSSIAN GOVERNMENT IDs ====================
    
    # Passport (Russian format: XX XX XXXXXX or XXXXXX)
    PASSPORT_PATTERNS = [
        # Series and number: 12 34 567890 or 1234567890
        r'\b\d{2}[\s-]?\d{2}[\s-]?\d{6}\b',
        # Just 10 digits (number only)
        r'\b\d{10}\b',
    ]
    
    # SNILS (Social insurance account: XXX-XXX-XXX XX or similar)
    SNILS_PATTERNS = [
        r'\b\d{3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2}\b',
    ]
    
    # INN (Russian tax ID: 10 or 12 digits)
    INN_PATTERNS = [
        r'\b\d{10}\b',  # 10-digit INN (individual)
        r'\b\d{12}\b',  # 12-digit INN (organization)
    ]
    
    # KPP (Russian tax reason code: 9 digits, often XXXXXX000)
    KPP_PATTERNS = [
        r'\b\d{9}\b',
    ]
    
    # OGRN (Russian registration number: 13 digits for LLC, 15 for individual)
    OGRN_PATTERNS = [
        r'\b\d{13}\b',  # OGRN
        r'\b\d{15}\b',  # OGRNIP
    ]
    
    # BIC (Bank identification code: 9 digits)
    BIC_PATTERNS = [
        r'\b\d{9}\b',
    ]
    
    # Driver's license (Russian format: XXXX XXXXXX or similar)
    DRIVER_LICENSE_PATTERNS = [
        r'\b(?:\d{2}[\s-]?){3}\d{2,3}\b',  # Series and number with separators
    ]
    
    # Temporary ID (Temporary residence permit: variable format)
    TEMPORARY_ID_PATTERNS = [
        r'\b\d{2}\s*(?:No|№|N)\s*\d{6,7}\b',
    ]
    
    # Birth certificate (Свидетельство о рождении: 10-12 digits)
    BIRTH_CERTIFICATE_PATTERNS = [
        r'\b\d{10,12}\b',
    ]
    
    # Residence permit series and number
    RESIDENCE_PERMIT_PATTERNS = [
        r'\b[A-Z]{2}\d{7}\b',  # Series (2 letters) and number
    ]
    
    # ==================== ONE-TIME CODES ====================
    
    # OTP / SMS codes (4-6 digits, often preceded by keywords)
    OTP_PATTERNS = [
        r'(?:code|код|otp|sms)\s*[:=]?\s*\d{4,6}',
    ]
    
    # ==================== PHYSICAL ADDRESSES ====================
    
    # Russian postal code (6 digits)
    POSTAL_CODE_PATTERNS = [
        r'\b\d{6}\b',
    ]
    
    @classmethod
    def compile_patterns(cls) -> Dict[str, List[Pattern]]:
        """
        Compile all patterns into regex Pattern objects.
        
        Returns:
            Dictionary mapping category name to list of compiled patterns
        """
        patterns = {}
        
        # Add all patterns from class attributes
        for attr_name in dir(cls):
            if attr_name.endswith('_PATTERNS') and attr_name.isupper():
                category_name = attr_name.replace('_PATTERNS', '')
                pattern_list = getattr(cls, attr_name)
                
                compiled = []
                for pattern_str in pattern_list:
                    try:
                        compiled.append(re.compile(pattern_str, re.IGNORECASE | re.UNICODE))
                    except re.error as e:
                        logger.warning(f"Failed to compile pattern '{pattern_str}' for {category_name}: {e}")
                        continue
                
                if compiled:
                    patterns[category_name] = compiled
        
        return patterns
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all supported categories."""
        patterns = cls.compile_patterns()
        return sorted(list(patterns.keys()))


def create_pattern_registry() -> Dict[str, List[Pattern]]:
    """Create and return compiled pattern registry."""
    return RegexPatterns.compile_patterns()


# Lazy-loaded global registry
_PATTERN_REGISTRY = None


def get_pattern_registry() -> Dict[str, List[Pattern]]:
    """Get global pattern registry (lazy-loaded)."""
    global _PATTERN_REGISTRY
    if _PATTERN_REGISTRY is None:
        _PATTERN_REGISTRY = create_pattern_registry()
    return _PATTERN_REGISTRY
