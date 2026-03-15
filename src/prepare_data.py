import pandas as pd
import ast
from typing import List, Tuple


def read_train_dataset(path: str):
    """Читаем тренировочный датасет (train_dataset.tsv)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["target"] = df["target"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else [])
    return df


def read_test_dataset(path: str):
    """Читаем тестовый датасет (private_test_dataset.csv)."""
    df = pd.read_csv(path, dtype=str)
    df["text"] = df["text"].astype(str)
    return df


def spans_to_bio_tags(text: str, spans: List[Tuple[int, int, str]]) -> List[str]:
    """
    Преобразуем символьные span'ы (start, end, label) в BIO-теги.
    
    ✅ ИСПРАВЛЕНИЕ: Обработка overlapping spans - берем последний (приоритет)
    """
    bio_tags = ["O"] * len(text)  # Инициализируем как "O"
    span_dict = {}
    
    # ✅ УЛУЧШЕНИЕ: Сортируем spans по (end-start) в обратном порядке
    # Это дает приоритет более длинным spans при пересечениях
    sorted_spans = sorted(spans, key=lambda x: (x[1] - x[0]), reverse=True)
    
    for start, end, label in sorted_spans:
        # Проверяем валидность span'а
        if start < 0 or end > len(text) or start >= end:
            continue
        
        for i in range(start, end):
            # Если позиция уже занята, пропускаем (был более длинный span)
            if i not in span_dict:
                span_dict[i] = label
    
    # Конвертируем span_dict в BIO-теги
    for i in range(len(text)):
        if i not in span_dict:
            bio_tags[i] = "O"
        else:
            label = span_dict[i]
            # B- если это начало span'а или отличается от предыдущего
            if i == 0 or span_dict.get(i - 1) != label:
                bio_tags[i] = f"B-{label}"
            else:
                bio_tags[i] = f"I-{label}"
    
    return bio_tags