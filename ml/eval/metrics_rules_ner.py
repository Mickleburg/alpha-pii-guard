"""
Unified metrics computation for alpha-pii-guard.

Implements strict span+category matching for regex, NER, and merged detectors.
"""

from typing import List, Tuple, Dict, Any, Set, Callable
from collections import defaultdict

from ml.entities import Entity, entity_to_tuple
from ml.merge.label_map import normalize_label


def to_normalized_set(
    spans: List[Tuple[int, int, str]] | List[Entity]
) -> Set[Tuple[int, int, str]]:
    """
    Convert span list to normalized set for comparison.
    
    Args:
        spans: List of tuples or Entity objects
        
    Returns:
        Set of (start, end, label) tuples with normalized labels
    """
    normalized = set()
    
    for item in spans:
        if isinstance(item, Entity):
            span = entity_to_tuple(item)
        else:
            span = item
        
        start, end, label = span
        normalized_label = normalize_label(label)
        normalized.add((start, end, normalized_label))
    
    return normalized


def compute_counts(
    pred: Set[Tuple[int, int, str]],
    gold: Set[Tuple[int, int, str]]
) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN counts for strict matching.
    
    Args:
        pred: Predicted entity set
        gold: Gold standard entity set
        
    Returns:
        Tuple of (tp, fp, fn)
    """
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, F1.
    
    Args:
        tp: True positive count
        fp: False positive count
        fn: False negative count
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_micro_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    references: List[List[Tuple[int, int, str]]]
) -> Dict[str, float]:
    """
    Compute micro-averaged metrics across all examples.
    
    Args:
        predictions: List of prediction span lists
        references: List of gold span lists
        
    Returns:
        Dict with precision, recall, f1 keys
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_spans, gold_spans in zip(predictions, references):
        pred_set = to_normalized_set(pred_spans)
        gold_set = to_normalized_set(gold_spans)
        
        tp, fp, fn = compute_counts(pred_set, gold_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_per_label_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    references: List[List[Tuple[int, int, str]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-category metrics.
    
    Args:
        predictions: List of prediction span lists
        references: List of gold span lists
        
    Returns:
        Dict mapping label to metrics dict
    """
    label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for pred_spans, gold_spans in zip(predictions, references):
        pred_set = to_normalized_set(pred_spans)
        gold_set = to_normalized_set(gold_spans)
        
        # TP by label
        for start, end, label in pred_set & gold_set:
            label_counts[label]["tp"] += 1
        
        # FP by label
        for start, end, label in pred_set - gold_set:
            label_counts[label]["fp"] += 1
        
        # FN by label
        for start, end, label in gold_set - pred_set:
            label_counts[label]["fn"] += 1
    
    # Compute metrics per label
    label_metrics = {}
    for label, counts in label_counts.items():
        precision, recall, f1 = precision_recall_f1(
            counts["tp"], counts["fp"], counts["fn"]
        )
        label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }
    
    return label_metrics


def evaluate_detector(
    detector_fn: Callable[[str], List[Tuple[int, int, str]]],
    texts: List[str],
    gold_spans: List[List[Tuple[int, int, str]]]
) -> Dict[str, Any]:
    """
    Evaluate detector function on dataset.
    
    Args:
        detector_fn: Function mapping text to list of spans
        texts: List of input texts
        gold_spans: List of gold span lists
        
    Returns:
        Dict with micro and per_label metrics
    """
    predictions = []
    
    for text in texts:
        pred = detector_fn(text)
        predictions.append(pred)
    
    micro = compute_micro_metrics(predictions, gold_spans)
    per_label = compute_per_label_metrics(predictions, gold_spans)
    
    return {
        "micro": micro,
        "per_label": per_label,
    }
