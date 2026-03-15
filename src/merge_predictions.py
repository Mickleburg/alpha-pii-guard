from typing import List, Tuple, Set

# ============================================================================
# СЛИЯНИЕ REGEX И NER ПРЕДСКАЗАНИЙ
# ============================================================================

def merge_predictions(
    regex_spans: List[Tuple[int, int, str]],
    ner_spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """
    Слияние regex (приоритет) и NER предсказаний с обработкой пересечений.
    
    Алгоритм:
    1. Regex добавляются в результат (высший приоритет)
    2. NER добавляются, если не пересекаются с regex
    3. Дубликаты удаляются
    4. Результат сортируется
    
    Args:
        regex_spans: Список spans из regex детектора
        ner_spans: Список spans из NER модели
    
    Returns:
        Объединённый список spans
    """
    if not regex_spans and not ner_spans:
        return []
    
    # Конвертируем в set для дедупликации
    result_set: Set[Tuple[int, int, str]] = set()
    
    # 1. Добавляем все regex spans (они имеют приоритет)
    for span in regex_spans:
        result_set.add(span)
    
    # 2. Добавляем NER spans, если они не пересекаются с regex
    for ner_start, ner_end, ner_label in ner_spans:
        # Проверяем пересечение с любым regex span
        overlaps_with_regex = False
        for regex_start, regex_end, regex_label in regex_spans:
            if _spans_overlap((ner_start, ner_end), (regex_start, regex_end)):
                overlaps_with_regex = True
                break
        
        # Если нет пересечения, добавляем
        if not overlaps_with_regex:
            result_set.add((ner_start, ner_end, ner_label))
    
    # 3. Конвертируем обратно в список и сортируем
    result = sorted(list(result_set), key=lambda x: (x[0], x[1]))
    
    return result


def _spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Проверяет, пересекаются ли два span'а.
    
    Args:
        span1: Первый span (start, end)
        span2: Второй span (start, end)
    
    Returns:
        True если пересекаются, иначе False
    """
    start1, end1 = span1
    start2, end2 = span2
    
    # Spans пересекаются, если конец одного > начало другого
    # И конец другого > начало одного
    return not (end1 <= start2 or end2 <= start1)


def merge_multiple(
    predictions: List[List[Tuple[int, int, str]]],
    weights: List[float] = None
) -> List[Tuple[int, int, str]]:
    """
    Слияние нескольких списков предсказаний (для ансамблей).
    
    Args:
        predictions: Список списков spans из разных моделей
        weights: Веса для каждой модели (опционально)
    
    Returns:
        Объединённый список spans
    """
    if not predictions:
        return []
    
    # Если одно предсказание - вернуть как есть
    if len(predictions) == 1:
        return sorted(predictions[0], key=lambda x: (x[0], x[1]))
    
    # Слияние двух первых, потом с остальными
    result = predictions[0]
    for pred in predictions[1:]:
        result = merge_predictions(result, pred)
    
    return result


def deduplicate_spans(
    spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """
    Удаляет дубликаты из списка spans.
    
    Args:
        spans: Список spans
    
    Returns:
        Список spans без дубликатов
    """
    return sorted(list(set(spans)), key=lambda x: (x[0], x[1]))


def merge_overlapping_spans(
    spans: List[Tuple[int, int, str]],
    strategy: str = "union"
) -> List[Tuple[int, int, str]]:
    """
    Слияние перекрывающихся spans.
    
    Args:
        spans: Список spans
        strategy: "union" (объединение) или "intersection" (пересечение)
    
    Returns:
        Обработанный список spans
    """
    if not spans:
        return []
    
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    result = []
    current = list(spans[0])
    
    for i in range(1, len(spans)):
        next_span = spans[i]
        
        if _spans_overlap((current[0], current[1]), (next_span[0], next_span[1])):
            if strategy == "union":
                # Расширяем текущий span
                current[0] = min(current[0], next_span[0])
                current[1] = max(current[1], next_span[1])
                # Объединяем метки
                if current[2] != next_span[2]:
                    current[2] = f"{current[2]}/{next_span[2]}"
            # Для "intersection" просто пропускаем
        else:
            result.append(tuple(current))
            current = list(next_span)
    
    result.append(tuple(current))
    return result


def filter_by_confidence(
    spans: List[Tuple[int, int, str]],
    confidence_scores: List[float] = None,
    threshold: float = 0.5
) -> List[Tuple[int, int, str]]:
    """
    Фильтрует spans по уровню доверия.
    
    Args:
        spans: Список spans
        confidence_scores: Баллы доверия для каждого span
        threshold: Минимальный порог доверия
    
    Returns:
        Отфильтрованный список spans
    """
    if confidence_scores is None or len(confidence_scores) != len(spans):
        return spans
    
    result = []
    for span, score in zip(spans, confidence_scores):
        if score >= threshold:
            result.append(span)
    
    return sorted(result, key=lambda x: (x[0], x[1]))