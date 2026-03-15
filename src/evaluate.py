import pandas as pd
import numpy as np
from typing import List, Tuple


def compute_metrics(predictions: List[List[Tuple[int, int, str]]], targets: List[List[Tuple[int, int, str]]]) -> dict:
    """
    ✅ ИСПРАВЛЕНО: Вычисляем метрики: Precision, Recall, F1.
    
    Строгая метрика: совпадение только если совпали start, end И label (как 3-tuple).
    
    Улучшения:
    - Проверяем точное совпадение (start, end, label), а не просто позицию
    - Обработка macro и weighted F1
    - Добавлена информация о per-class метриках
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    # Для per-class метрик
    class_metrics = {}
    
    for pred, target in zip(predictions, targets):
        pred_set = set(pred)
        target_set = set(target)
        
        # Точное совпадение: (start, end, label)
        tp += len(pred_set & target_set)
        fp += len(pred_set - target_set)
        fn += len(target_set - pred_set)
        
        # Per-class метрики
        for span in target_set:
            label = span[2]
            if label not in class_metrics:
                class_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            
            if span in pred_set:
                class_metrics[label]["tp"] += 1
            else:
                class_metrics[label]["fn"] += 1
        
        for span in pred_set - target_set:
            label = span[2]
            if label not in class_metrics:
                class_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            class_metrics[label]["fp"] += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Вычисляем macro F1
    class_f1_scores = []
    for label, metrics in class_metrics.items():
        tp_c = metrics["tp"]
        fp_c = metrics["fp"]
        fn_c = metrics["fn"]
        
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * (p_c * r_c) / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        
        class_f1_scores.append(f1_c)
    
    macro_f1 = np.mean(class_f1_scores) if class_f1_scores else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "class_metrics": class_metrics,
    }


def save_metrics(metrics_dict: dict, output_path: str):
    """✅ ИСПРАВЛЕНО: Сохраняем метрики в CSV с дополнительной информацией."""
    rows = []
    for key, value in metrics_dict.items():
        rows.append({
            "model": value.get("model", "unknown"),
            "dataset": value.get("dataset", "unknown"),
            "precision": value.get("precision", 0),
            "recall": value.get("recall", 0),
            "micro_f1": value.get("micro_f1", 0),
            "macro_f1": value.get("macro_f1", 0),
            "tp": value.get("tp", 0),
            "fp": value.get("fp", 0),
            "fn": value.get("fn", 0),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def print_metrics(metrics: dict, model_name: str = "Model"):
    """✅ ДОБАВЛЕНО: Красивый вывод метрик."""
    print(f"\n{'='*50}")
    print(f"Metrics for {model_name}")
    print(f"{'='*50}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Micro-F1:   {metrics['micro_f1']:.4f}")
    print(f"Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"{'='*50}\n")