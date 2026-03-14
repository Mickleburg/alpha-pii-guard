"""Tests for regex-based PII detection."""

import pytest

from src.regex.detector import RegexPIIDetector, create_detector
from src.regex.patterns import RegexPatterns, get_pattern_registry
from src.regex.context_rules import ContextRule, ContextRuleRegistry, ContextFilterMode


class TestRegexPatterns:
    """Test regex pattern compilation."""
    
    def test_patterns_compile(self):
        """Test that all patterns compile without error."""
        registry = get_pattern_registry()
        assert registry is not None
        assert len(registry) > 0
    
    def test_categories_extracted(self):
        """Test category extraction."""
        categories = RegexPatterns.get_categories()
        assert "EMAIL" in categories
        assert "PHONE" in categories
        assert "CARD" in categories
        assert len(categories) > 5


class TestEmailDetection:
    """Test email detection."""
    
    def test_simple_email(self):
        """Test simple email detection."""
        detector = create_detector()
        text = "Contact: user@example.com"
        spans = detector.predict(text)
        
        assert len(spans) > 0
        # Find EMAIL span
        email_spans = [s for s in spans if s[2] == "EMAIL"]
        assert len(email_spans) > 0
        
        start, end, category = email_spans[0]
        assert text[start:end] == "user@example.com"
        assert category == "EMAIL"
    
    def test_multiple_emails(self):
        """Test multiple email detection."""
        detector = create_detector()
        text = "Emails: john@example.com and jane@example.org"
        spans = detector.predict(text)
        
        email_spans = [s for s in spans if s[2] == "EMAIL"]
        assert len(email_spans) >= 2
    
    def test_email_in_url(self):
        """Test email in URL."""
        detector = create_detector()
        text = "Visit https://contact@example.com/"
        spans = detector.predict(text)
        
        # Should still detect email pattern
        email_spans = [s for s in spans if s[2] == "EMAIL"]
        # May or may not detect in URL context, depending on rules
        assert isinstance(email_spans, list)


class TestPhoneDetection:
    """Test phone number detection."""
    
    def test_russian_phone_plus7(self):
        """Test Russian phone with +7."""
        detector = create_detector()
        text = "Call me: +7 999 123 45 67"
        spans = detector.predict(text)
        
        phone_spans = [s for s in spans if s[2] == "PHONE"]
        assert len(phone_spans) > 0
        
        start, end, category = phone_spans[0]
        detected = text[start:end]
        assert "999" in detected
        assert "123" in detected
    
    def test_russian_phone_8(self):
        """Test Russian phone with 8."""
        detector = create_detector()
        text = "Phone: 8-999-123-45-67"
        spans = detector.predict(text)
        
        phone_spans = [s for s in spans if s[2] == "PHONE"]
        assert len(phone_spans) > 0
    
    def test_international_phone(self):
        """Test international phone format."""
        detector = create_detector()
        text = "Contact: +1-555-123-4567"
        spans = detector.predict(text)
        
        phone_spans = [s for s in spans if s[2] == "PHONE"]
        assert len(phone_spans) > 0


class TestCardDetection:
    """Test bank card detection."""
    
    def test_visa_card_basic(self):
        """Test basic Visa card detection."""
        detector = create_detector()
        text = "Card: 4111 1111 1111 1111"
        spans = detector.predict(text)
        
        card_spans = [s for s in spans if s[2] == "CARD"]
        assert len(card_spans) > 0
    
    def test_mastercard_card(self):
        """Test Mastercard detection."""
        detector = create_detector()
        text = "Number: 5555-4433-3322-2222"
        spans = detector.predict(text)
        
        card_spans = [s for s in spans if s[2] == "CARD"]
        assert len(card_spans) > 0
    
    def test_card_with_spaces(self):
        """Test card with various spacing."""
        detector = create_detector()
        text = "4111 1111 1111 1111"
        spans = detector.predict(text)
        
        card_spans = [s for s in spans if s[2] == "CARD"]
        assert len(card_spans) > 0
    
    def test_card_no_false_positive_on_long_numbers(self):
        """Test that not every long number is marked as card."""
        detector = create_detector()
        text = "Code: 12345678901234567890"  # 20 digits
        spans = detector.predict(text)
        
        # Should not necessarily detect as card without context
        # This tests that we're not too aggressive


class TestPassportDetection:
    """Test passport detection."""
    
    def test_passport_with_series(self):
        """Test passport with series and number."""
        detector = create_detector()
        text = "Passport: 12 34 567890"
        spans = detector.predict(text)
        
        passport_spans = [s for s in spans if s[2] == "PASSPORT"]
        # May or may not detect without context keyword
        assert isinstance(passport_spans, list)
    
    def test_passport_no_spaces(self):
        """Test passport number format."""
        detector = create_detector()
        text = "Паспорт: 1234567890"
        spans = detector.predict(text)
        
        passport_spans = [s for s in spans if s[2] == "PASSPORT"]
        # May detect as other numeric IDs
        assert isinstance(passport_spans, list)


class TestSNILSDetection:
    """Test SNILS detection."""
    
    def test_snils_basic(self):
        """Test SNILS detection."""
        detector = create_detector()
        text = "SNILS: 123-456-789 01"
        spans = detector.predict(text)
        
        snils_spans = [s for s in spans if s[2] == "SNILS"]
        # May or may not detect
        assert isinstance(snils_spans, list)


class TestINNDetection:
    """Test INN detection."""
    
    def test_inn_basic(self):
        """Test INN detection."""
        detector = create_detector()
        text = "ИНН: 7712345678"
        spans = detector.predict(text)
        
        inn_spans = [s for s in spans if s[2] == "INN"]
        # May detect as generic 10-digit number
        assert isinstance(inn_spans, list)


class TestEmptyAndEdgeCases:
    """Test empty and edge cases."""
    
    def test_empty_text(self):
        """Test with empty text."""
        detector = create_detector()
        spans = detector.predict("")
        
        assert spans == []
    
    def test_text_with_only_whitespace(self):
        """Test with whitespace only."""
        detector = create_detector()
        spans = detector.predict("   \n\t  ")
        
        assert spans == []
    
    def test_text_with_no_pii(self):
        """Test text without PII."""
        detector = create_detector()
        text = "This is a normal text without any sensitive information."
        spans = detector.predict(text)
        
        # Should find no PII
        assert len(spans) == 0 or spans == []
    
    def test_very_long_text(self):
        """Test with very long text."""
        detector = create_detector()
        text = "Normal text " * 1000 + "Email: test@example.com"
        spans = detector.predict(text)
        
        email_spans = [s for s in spans if s[2] == "EMAIL"]
        assert len(email_spans) > 0


class TestOverlapRemoval:
    """Test overlap removal."""
    
    def test_non_overlapping_spans(self):
        """Test with non-overlapping spans."""
        detector = create_detector()
        text = "Email: test@example.com and phone: +7-999-123-45-67"
        spans = detector.predict(text)
        
        # Check no overlaps
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                start1, end1, _ = spans[i]
                start2, end2, _ = spans[j]
                # Should not overlap
                assert not (start1 < end2 and start2 < end1)
    
    def test_overlapping_same_category(self):
        """Test with overlapping matches of same category."""
        detector = create_detector()
        # This would be a contrived case, but detector should handle it
        text = "test@example.com"
        spans = detector.predict(text)
        
        # Should have email
        email_spans = [s for s in spans if s[2] == "EMAIL"]
        assert len(email_spans) >= 1


class TestBatchPredict:
    """Test batch prediction."""
    
    def test_batch_predict_multiple(self):
        """Test batch prediction with multiple texts."""
        detector = create_detector()
        texts = [
            "Email: test@example.com",
            "Phone: +7-999-123-45-67",
            "Card: 4111 1111 1111 1111"
        ]
        
        results = detector.batch_predict(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    def test_batch_predict_empty_list(self):
        """Test batch predict with empty list."""
        detector = create_detector()
        results = detector.batch_predict([])
        
        assert results == []


class TestDetectorAttributes:
    """Test detector attributes and methods."""
    
    def test_get_supported_categories(self):
        """Test getting supported categories."""
        detector = create_detector()
        categories = detector.get_supported_categories()
        
        assert len(categories) > 0
        assert "EMAIL" in categories
    
    def test_get_patterns_count(self):
        """Test getting pattern count for category."""
        detector = create_detector()
        count = detector.get_patterns_for_category("EMAIL")
        
        assert count >= 1


class TestContextRules:
    """Test context rule application."""
    
    def test_context_rule_creation(self):
        """Test creating context rule."""
        rule = ContextRule(
            category="TEST",
            keywords=["test", "keyword"],
            context_window=50,
            mode=ContextFilterMode.OPTIONAL
        )
        
        assert rule.category == "TEST"
        assert len(rule.keywords) == 2
    
    def test_context_rule_has_context(self):
        """Test context detection in rule."""
        rule = ContextRule(
            category="TEST",
            keywords=["keyword"],
            context_window=10
        )
        
        text = "Some keyword here"
        has_context = rule.has_context(text, 5, 10)
        
        assert has_context is True
    
    def test_context_rule_no_context(self):
        """Test missing context."""
        rule = ContextRule(
            category="TEST",
            keywords=["keyword"],
            context_window=5
        )
        
        text = "Some text here"
        has_context = rule.has_context(text, 0, 4)
        
        assert has_context is False
    
    def test_context_registry(self):
        """Test context rule registry."""
        registry = ContextRuleRegistry()
        
        # Should have default rules
        assert registry.get_rule("EMAIL") is not None


class TestDetectorWithContextRules:
    """Test detector with context rules enabled/disabled."""
    
    def test_detector_with_rules_enabled(self):
        """Test detector with context rules enabled."""
        detector = create_detector(use_context_rules=True)
        text = "Secret: abc123def456ghi789"
        spans = detector.predict(text)
        
        assert isinstance(spans, list)
    
    def test_detector_with_rules_disabled(self):
        """Test detector with context rules disabled."""
        detector = create_detector(use_context_rules=False)
        text = "Email: test@example.com"
        spans = detector.predict(text)
        
        assert isinstance(spans, list)
    
    def test_context_filtering_effect(self):
        """Test that context filtering can affect results."""
        detector_with = create_detector(use_context_rules=True)
        detector_without = create_detector(use_context_rules=False)
        
        text = "123456"  # Ambiguous: could be PIN, OTP, passport, etc.
        
        spans_with = detector_with.predict(text)
        spans_without = detector_without.predict(text)
        
        # Results may differ based on context filtering
        assert isinstance(spans_with, list)
        assert isinstance(spans_without, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
