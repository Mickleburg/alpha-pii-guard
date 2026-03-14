"""Context-based filtering rules for regex detections."""

import re
from typing import List, Tuple, Optional, Set
from enum import Enum

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ContextFilterMode(Enum):
    """Mode for context filtering."""
    ALLOW = "allow"  # Only match if context keyword found
    DENY = "deny"    # Skip match if context keyword found
    OPTIONAL = "optional"  # Match regardless, but boost confidence if context found


class ContextRule:
    """Rule for context-based filtering."""
    
    def __init__(
        self,
        category: str,
        keywords: List[str],
        context_window: int = 50,
        mode: ContextFilterMode = ContextFilterMode.OPTIONAL
    ):
        """
        Initialize context rule.
        
        Args:
            category: PII category
            keywords: Keywords indicating this PII type (case-insensitive)
            context_window: Characters to check before/after match
            mode: Filtering mode
        """
        self.category = category
        self.keywords = [kw.lower() for kw in keywords]
        self.context_window = context_window
        self.mode = mode
    
    def has_context(self, text: str, start: int, end: int) -> bool:
        """
        Check if match has appropriate context keywords.
        
        Args:
            text: Full text
            start: Match start position
            end: Match end position
            
        Returns:
            True if context keyword found in window
        """
        # Extract context window
        window_start = max(0, start - self.context_window)
        window_end = min(len(text), end + self.context_window)
        context = text[window_start:window_end].lower()
        
        # Check for any keyword
        for keyword in self.keywords:
            if keyword in context:
                return True
        
        return False
    
    def should_accept(self, text: str, start: int, end: int) -> bool:
        """
        Determine if match should be accepted based on context.
        
        Args:
            text: Full text
            start: Match start position
            end: Match end position
            
        Returns:
            True if match should be accepted
        """
        has_context = self.has_context(text, start, end)
        
        if self.mode == ContextFilterMode.ALLOW:
            # Only accept if context found
            return has_context
        elif self.mode == ContextFilterMode.DENY:
            # Accept only if context NOT found
            return not has_context
        else:  # OPTIONAL
            # Always accept
            return True


class ContextRuleRegistry:
    """Registry of context rules for different categories."""
    
    def __init__(self):
        """Initialize registry with default rules."""
        self.rules = {}
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default context rules."""
        
        # API Key: usually preceded by "api", "key", "secret", etc.
        self.add_rule(ContextRule(
            category="API_KEY",
            keywords=["api", "key", "secret", "token", "credential"],
            context_window=20,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # CVV/CVC: usually preceded by security-related keywords
        self.add_rule(ContextRule(
            category="CVV_CVC",
            keywords=["cvv", "cvc", "cvx", "security", "code", "back"],
            context_window=10,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # PIN: usually preceded by PIN keyword
        self.add_rule(ContextRule(
            category="PIN",
            keywords=["pin", "пин"],
            context_window=10,
            mode=ContextFilterMode.ALLOW
        ))
        
        # Password: usually preceded by password keyword
        self.add_rule(ContextRule(
            category="PASSWORD",
            keywords=["password", "пароль", "pass"],
            context_window=10,
            mode=ContextFilterMode.ALLOW
        ))
        
        # Email: no strict context needed
        self.add_rule(ContextRule(
            category="EMAIL",
            keywords=[],
            context_window=0,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # Phone: usually preceded by phone-related keywords
        self.add_rule(ContextRule(
            category="PHONE",
            keywords=["phone", "tel", "call", "номер", "телефон"],
            context_window=20,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # Card: usually preceded by card-related keywords
        self.add_rule(ContextRule(
            category="CARD",
            keywords=["card", "карта", "credit", "visa", "mastercard", "amex", "number"],
            context_window=30,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # Passport: usually preceded by passport-related keywords
        self.add_rule(ContextRule(
            category="PASSPORT",
            keywords=["passport", "паспорт", "series", "series and number"],
            context_window=30,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # SNILS: usually preceded by SNILS keyword
        self.add_rule(ContextRule(
            category="SNILS",
            keywords=["snils", "снилс", "insurance", "страховой"],
            context_window=20,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # INN: usually preceded by INN keyword
        self.add_rule(ContextRule(
            category="INN",
            keywords=["inn", "инн", "tax id", "налоговый"],
            context_window=20,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # Account: usually preceded by account keyword
        self.add_rule(ContextRule(
            category="ACCOUNT",
            keywords=["account", "счет", "расчетный", "settlement"],
            context_window=30,
            mode=ContextFilterMode.OPTIONAL
        ))
        
        # OTP: usually preceded by code keyword
        self.add_rule(ContextRule(
            category="OTP",
            keywords=["code", "otp", "sms", "код", "sms code"],
            context_window=10,
            mode=ContextFilterMode.OPTIONAL
        ))
    
    def add_rule(self, rule: ContextRule) -> None:
        """Add or update a context rule."""
        self.rules[rule.category] = rule
    
    def get_rule(self, category: str) -> Optional[ContextRule]:
        """Get rule for category."""
        return self.rules.get(category)
    
    def should_accept(self, category: str, text: str, start: int, end: int) -> bool:
        """
        Check if match should be accepted according to rules.
        
        Args:
            category: PII category
            text: Full text
            start: Match start
            end: Match end
            
        Returns:
            True if match should be accepted
        """
        rule = self.get_rule(category)
        if rule is None:
            # No rule = accept by default
            return True
        
        return rule.should_accept(text, start, end)


# Global rule registry (lazy-loaded)
_RULE_REGISTRY = None


def get_rule_registry() -> ContextRuleRegistry:
    """Get global rule registry (lazy-loaded)."""
    global _RULE_REGISTRY
    if _RULE_REGISTRY is None:
        _RULE_REGISTRY = ContextRuleRegistry()
    return _RULE_REGISTRY
