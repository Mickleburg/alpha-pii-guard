import re
from typing import Dict, List, Tuple, Optional

# ============================================================================
# ЧАСТЬ 1: REGEX ПАТТЕРНЫ + КОНТЕКСТНЫЕ ПРАВИЛА
# ============================================================================

PII_PATTERNS = {
    "Номер телефона": re.compile(
        r"\b(?:\+?7|8)[-\s]?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{2}\b"
    ),
    "Email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "Паспортные данные": re.compile(
        r"\b(?:[0-9]{4}[-\s]?[0-9]{6}|[0-9]{2}[-\s][0-9]{2}[-\s][0-9]{6})\b"
    ),
    "Сведения об ИНН": re.compile(
        r"\b(?:[0-9]{10}|[0-9]{12})\b"
    ),
    "СНИЛС клиента": re.compile(
        r"\b[0-9]{3}-[0-9]{3}-[0-9]{3}\s[0-9]{2}\b"
    ),
    "Номер карты": re.compile(
        r"\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b"
    ),
    "CVV/CVC": re.compile(
        r"\b[0-9]{3}\b"
    ),
    "Номер банковского счета": re.compile(
        r"\b[0-9]{20}\b"
    ),
    "Водительское удостоверение": re.compile(
        r"\b[0-9]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{6}\b"
    ),
    "Временное удостоверение личности": re.compile(
        r"\b[IVXLCDM]{1,4}[-\s]?[А-Я]{2}[-\s]?[0-9]{6}\b"
    ),
    "Серия и номер вида на жительство": re.compile(
        r"\b[0-9]{2}[-\s]?[0-9]{7}\b"
    ),
    "Свидетельство о рождении": re.compile(
        r"\b[IVXLCDM]{1,4}-[А-Я]{2}[-\s]?[0-9]{6}\b"
    ),
    "ПИН код": re.compile(
        r"\b[0-9]{4}\b"
    ),
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)": re.compile(
        r"\b(?:[0-9]{10}|[0-9]{9}|[0-9]{13})\b"
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
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
]

# ============================================================================
# КОНТЕКСТНЫЕ КЛЮЧЕВЫЕ СЛОВА
# ============================================================================

CONTEXT_KEYWORDS: Dict[str, List[str]] = {
    "CVV/CVC": ["cvv", "cvc", "cvv2", "cvc2", "секретный", "код безопасности"],
    "Номер карты": ["карт", "card", "номер карты", "номер счета"],
    "Номер телефона": ["телефон", "сбп", "смс", "звон", "номер", "контакт", "мобильный"],
    "Email": ["email", "e-mail", "почт", "письм", "адрес электронной"],
    "ПИН код": ["пин", "pin", "код доступа", "пароль"],
    "Номер банковского счета": ["счет", "расчетн", "банковск", "номер счета"],
    "Паспортные данные": ["паспорт", "серия", "выдан", "мвд", "уфмс", "подразделени"],
    "Сведения об ИНН": ["инн", "налогов", "ип", "ооо"],
    "СНИЛС клиента": ["снилс", "страхов", "пенсион"],
    "Водительское удостоверение": ["водител", "ву", "прав", "удостовер"],
    "Временное удостоверение личности": ["временн", "справк"],
    "Вид на жительство": ["внж", "вид на", "жительств"],
    "Свидетельство о рождении": ["свидетельств", "ребенк"],
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)": [
        "огрн", "кпп", "бик", "юр", "организац", "ооо", "ао", "ип", "компани"
    ],
}

# ============================================================================
# ЧАСТЬ 2: ФУНКЦИИ ДЕТЕКЦИИ
# ============================================================================

def detect_pii(text: str, context_radius: int = 30) -> List[Tuple[int, int, str]]:
    """
    Детектит PII в тексте с использованием regex паттернов и контекстных правил.
    
    Args:
        text: Входной текст для анализа
        context_radius: Радиус поиска контекстных ключевых слов
    
    Returns:
        Отсортированный список кортежей (start, end, label)
    """
    if not text:
        return []
    
    candidates = []
    
    # Проходим по всем паттернам
    for pattern_name in RULE_ONLY_ENTITIES:
        if pattern_name not in PII_PATTERNS:
            continue
            
        pattern = PII_PATTERNS[pattern_name]
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            
            # Проверяем контекст
            if _has_context_keyword(text, start, end, context_radius, pattern_name):
                candidates.append((start, end, pattern_name))
    
    # Удаляем перекрытия и возвращаем
    return sorted(_remove_overlaps(candidates), key=lambda x: x[0])


def _has_context_keyword(text: str, start: int, end: int, radius: int, label: str) -> bool:
    """
    Проверяет наличие контекстного ключевого слова.
    
    Args:
        text: Исходный текст
        start: Начало совпадения
        end: Конец совпадения
        radius: Радиус поиска контекста
        label: Метка типа PII
    
    Returns:
        True если найдено ключевое слово, иначе False
    """
    context_start = max(0, start - radius)
    context_end = min(len(text), end + radius)
    context = text[context_start:context_end].lower()
    
    keywords = CONTEXT_KEYWORDS.get(label, [])
    for keyword in keywords:
        if keyword.lower() in context:
            return True
    
    return False


def _remove_overlaps(candidates: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Убирает пересекающиеся кандидаты, оставляя первый (по старшинству).
    
    Args:
        candidates: Список кандидатов (start, end, label)
    
    Returns:
        Отфильтрованный список без пересечений
    """
    if not candidates:
        return []
    
    candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    result = [candidates[0]]
    
    for curr in candidates[1:]:
        # Если текущий не пересекается с последним в результате
        if curr[0] >= result[-1][1]:
            result.append(curr)
    
    return result


def find_all(text: str, patterns: Dict[str, re.Pattern]) -> List[Tuple[int, int, str]]:
    """
    Находит все совпадения для заданных паттернов.
    
    Args:
        text: Входной текст
        patterns: Словарь паттернов {имя: compiled regex}
    
    Returns:
        Список найденных совпадений
    """
    matches = []
    for name, pattern in patterns.items():
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), name))
    return matches


# Для обратной совместимости
detect_by_rules = detect_pii


# ============================================================================
# ЧАСТЬ 3: УТИЛИТЫ
# ============================================================================

def get_all_patterns() -> Dict[str, re.Pattern]:
    """Возвращает все доступные паттерны."""
    return PII_PATTERNS.copy()


def get_rule_only_entities() -> List[str]:
    """Возвращает список сущностей для rule-based детекции."""
    return RULE_ONLY_ENTITIES.copy()


def get_context_keywords() -> Dict[str, List[str]]:
    """Возвращает словарь контекстных ключевых слов."""
    return CONTEXT_KEYWORDS.copy()