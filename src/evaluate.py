import pandas as pd
from typing import List, Tuple

def compute_metrics(predictions: List[List[Tuple[int, int, str]]], targets: List[List[Tuple[int, int, str]]]) -> dict:
    """
    Вычисляем метрики: Precision, Recall, F1.
    Строгая метрика: совпадение только если совпали start, end И label.
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    for pred, target in zip(predictions, targets):
        pred_set = set(pred)
        target_set = set(target)
        
        tp += len(pred_set & target_set)
        fp += len(pred_set - target_set)
        fn += len(target_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

def save_metrics(metrics_dict: dict, output_path: str):
    """Сохраняем метрики в CSV."""
    rows = []
    for key, value in metrics_dict.items():
        rows.append({
            "model": value.get("model", "unknown"),
            "dataset": value.get("dataset", "unknown"),
            "precision": value.get("precision", 0),
            "recall": value.get("recall", 0),
            "micro_f1": value.get("micro_f1", 0),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")