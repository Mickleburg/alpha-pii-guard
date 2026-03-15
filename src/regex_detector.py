import re
from typing import Dict, List, Tuple

PII_PATTERNS = {
    "Номер телефона": re.compile(
        r"\b(?:\+?7|8)[\s\-]?\(?([0-9]{3})\)?[\s\-]?([0-9]{3})[\s\-]?([0-9]{2})[\s\-]?([0-9]{2})\b"
    ),
    "Email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "Паспортные данные": re.compile(
        r"\b[0-9]{4}[\s\-]?[0-9]{6}\b"
    ),
    "Сведения об ИНН": re.compile(
        r"\b(?:[0-9]{10}|[0-9]{12})\b"
    ),
    "СНИЛС клиента": re.compile(
        r"\b[0-9]{3}[\s\-][0-9]{3}[\s\-][0-9]{3}[\s\-]?[0-9]{2}\b"
    ),
    "Номер карты": re.compile(
        r"\b(?:[0-9]{4}[\s\-]?){3}[0-9]{4}\b"
    ),
    "CVV/CVC": re.compile(
        r"\b[0-9]{3,4}\b"
    ),
    "Номер банковского счета": re.compile(
        r"\b[0-9]{20}\b"
    ),
    "Водительское удостоверение": re.compile(
        r"\b[0-9]{2}[\s\-]?[0-9]{2}[\s\-]?[0-9]{6}\b"
    ),
    "Временное удостоверение личности": re.compile(
        r"\b[IVXLCDM]{1,4}[\s\-]?[А-Я]{2}[\s\-]?[0-9]{6}\b"
    ),
    "Серия и номер вида на жительство": re.compile(
        r"\b[0-9]{2}[\s\-]?[0-9]{7}\b"
    ),
    "Свидетельство о рождении": re.compile(
        r"\b[IVXLCDM]{1,4}[\s\-]?[А-Я]{2}[\s\-]?[0-9]{6}\b"
    ),
    "ПИН код": re.compile(
        r"\b[0-9]{4}\b"
    ),
    "Дата рождения": re.compile(
        r"\b(?:[0-3]?[0-9][./-]?){2}[12][90][0-9]{2}\b"
    ),
    "API ключи": re.compile(
        r"\b[a-zA-Z0-9_]{32,}\b"
    ),
}

RULE_ONLY_ENTITIES = [
    "Паспортные данные",
    "Сведения об ИНН",
    "СНИЛС клиента",
    "Водительское удостоверение",
    "Номер карты",
    "CVV/CVC",
    "Номер телефона",
    "Email",
    "Номер банковского счета",
    "Временное удостоверение личности",
    "Серия и номер вида на жительство",
    "Свидетельство о рождении",
    "ПИН код",
    "Дата рождения",
    "API ключи",
]

CONTEXT_KEYWORDS: Dict[str, List[str]] = {
    "CVV/CVC": ["cvv", "cvc", "cvc2", "секурити"],
    "Номер карты": ["карт", "card", "номер карты", "карты"],
    "Номер телефона": ["телефон", "сбп", "смс", "звон", "номер", "тел"],
    "Email": ["email", "e-mail", "почт", "письм"],
    "ПИН код": ["пин", "pin", "код"],
    "Номер банковского счета": ["счет", "расчетн", "банковск", "лс"],
    "Паспортные данные": ["паспорт", "серия", "выдан", "мвд", "уфмс"],
    "Сведения об ИНН": ["инн", "налогов"],
    "СНИЛС клиента": ["снилс", "страхов"],
    "Водительское удостоверение": ["водител", "ву", "прав", "в/у"],
    "API ключи": ["api", "key", "ключ"],
    "Дата рождения": ["рождени", "дата", "д.р.", "др.", "birthday", "born"],
}

def _has_context_keyword(text: str, start: int, end: int, radius: int, label: str) -> bool:
    context_start = max(0, start - radius)
    context_end = min(len(text), end + radius)
    context = text[context_start:context_end].lower()
    keywords = CONTEXT_KEYWORDS.get(label, [])
    for keyword in keywords:
        if keyword.lower() in context:
            return True
    return False

def _remove_overlaps(candidates: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    result = [candidates[0]]
    for curr in candidates[1:]:
        if curr[0] >= result[-1][1]:
            result.append(curr)
    return result

def detect_pii(text: str, context_radius: int = 50) -> List[Tuple[int, int, str]]:
    """Детектируем PII используя regex с контекстом."""
    if not text or len(text) == 0:
        return []
    
    candidates = []
    
    for pattern_name in RULE_ONLY_ENTITIES:
        if pattern_name not in PII_PATTERNS:
            continue
        pattern = PII_PATTERNS[pattern_name]
        try:
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                # Проверяем контекст для более точного детектирования
                if _has_context_keyword(text, start, end, context_radius, pattern_name):
                    candidates.append((start, end, pattern_name))
        except Exception as e:
            print(f"Error with pattern {pattern_name}: {e}")
            continue
    
    return sorted(_remove_overlaps(candidates), key=lambda x: x[0])