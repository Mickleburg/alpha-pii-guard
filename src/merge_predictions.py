from typing import List, Tuple
from src.utils import remove_overlaps

def merge_predictions(regex_spans: List[Tuple[int, int, str]], ner_spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Merge regex (приоритет) и NER predictions.
    Regex имеет приоритет - если regex нашла что-то, используем это.
    NER добавляем только если нет перекрытия с regex.
    """
    
    if not regex_spans and not ner_spans:
        return []
    
    result = list(regex_spans)
    
    for ner_start, ner_end, ner_label in ner_spans:
        overlaps = False
        for regex_start, regex_end, regex_label in regex_spans:
            # Проверяем пересечение
            if not (ner_end <= regex_start or ner_start >= regex_end):
                overlaps = True
                break
        if not overlaps:
            result.append((ner_start, ner_end, ner_label))
    
    # Удаляем перекрытия если вдруг они появились
    result = remove_overlaps(result)
    return sorted(result, key=lambda x: (x[0], x[1]))