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
    """Преобразуем символьные span'ы (start, end, label) в BIO-теги."""
    bio_tags = []
    span_dict = {}
    
    for start, end, label in spans:
        for i in range(start, end):
            span_dict[i] = label
    
    for i in range(len(text)):
        if i not in span_dict:
            bio_tags.append("O")
        else:
            label = span_dict[i]
            if i == 0 or span_dict.get(i - 1) != label:
                bio_tags.append(f"B-{label}")
            else:
                bio_tags.append(f"I-{label}")
    
    return bio_tags