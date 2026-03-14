"""
Merge priority configuration for alpha-pii-guard.

Defines which detector (regex or NER) takes priority for each entity category.
"""

from typing import Dict, Set


# Categories where regex-first makes sense (structured, rule-based)
REGEX_PRIORITY_LABELS: Set[str] = {
    "Email",
    "Номер телефона",
    "Паспортные данные",
    "Сведения об ИНН",
    "СНИЛС клиента",
    "Номер карты",
    "Номер банковского счета",
    "CVV/CVC",
    "ПИН код",
    "Дата окончания срока действия карты",
    "Одноразовые коды",
    "API ключи",
    "Серия и номер вида на жительство",
    "Водительское удостоверение",
    "Временное удостоверение личности",
    "Свидетельство о рождении",
    "Содержимое магнитной полосы",
}


# Categories where NER-first makes sense (context-dependent)
NER_PRIORITY_LABELS: Set[str] = {
    "ФИО",
    "Полный адрес",
    "Место рождения",
    "Гражданство и названия стран",
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
    "Наименование банка",
    "Имя держателя карты",
    "Данные об автомобиле клиента",
    "Разрешение на работу / визу",
    "Кодовые слова",
}


# Fallback priority order by source
DEFAULT_SOURCE_PRIORITY: Dict[str, int] = {
    "regex": 1,
    "ner": 2,
    "merged": 0,
}


# Category source priority mapping
CATEGORY_SOURCE_PRIORITY: Dict[str, str] = {}

for label in REGEX_PRIORITY_LABELS:
    CATEGORY_SOURCE_PRIORITY[label] = "regex"

for label in NER_PRIORITY_LABELS:
    CATEGORY_SOURCE_PRIORITY[label] = "ner"


# Structured labels (prefer shorter, more precise spans)
STRUCTURED_LABELS: Set[str] = REGEX_PRIORITY_LABELS.copy()


# Context labels (prefer longer spans for full context)
CONTEXT_LABELS: Set[str] = NER_PRIORITY_LABELS.copy()


def get_category_priority(label: str) -> str:
    """
    Get preferred detector source for given label.
    
    Args:
        label: Entity category name
        
    Returns:
        'regex' or 'ner' priority source, defaults to 'regex'
    """
    return CATEGORY_SOURCE_PRIORITY.get(label, "regex")


def is_structured_label(label: str) -> bool:
    """
    Check if label represents structured entity.
    
    Args:
        label: Entity category name
        
    Returns:
        True if structured (prefer shorter spans)
    """
    return label in STRUCTURED_LABELS


def is_context_label(label: str) -> bool:
    """
    Check if label represents context-dependent entity.
    
    Args:
        label: Entity category name
        
    Returns:
        True if context-dependent (prefer longer spans)
    """
    return label in CONTEXT_LABELS
