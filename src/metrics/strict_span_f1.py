"""Strict span + category F1 metric for PII NER evaluation."""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type alias for span tuple: (start, end, category)
SpanTuple = Tuple[int, int, str]


@dataclass
class MetricsResult:
    """Result of metric computation."""
    
    precision: float
    recall: float
    f1: float
    tp: int = 0  # True positives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives
    support: int = 0  # Total positives in ground truth
    
    def __repr__(self) -> str:
        return (
            f"MetricsResult(P={self.precision:.4f}, R={self.recall:.4f}, "
            f"F1={self.f1:.4f}, TP={self.tp}, FP={self.fp}, FN={self.fn})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support
        }


class StrictSpanMatcher:
    """Matcher for strict span and category matching."""
    
    @staticmethod
    def spans_equal(span1: SpanTuple, span2: SpanTuple) -> bool:
        """
        Check if two spans are exactly equal.
        
        Compares start, end, and category with strict equality.
        
        Args:
            span1: First span (start, end, category)
            span2: Second span (start, end, category)
            
        Returns:
            True if spans are exactly equal
        """
        if len(span1) != 3 or len(span2) != 3:
            return False
        
        start1, end1, cat1 = span1
        start2, end2, cat2 = span2
        
        return start1 == start2 and end1 == end2 and cat1 == cat2
    
    @staticmethod
    def find_matches(
        y_true: List[SpanTuple],
        y_pred: List[SpanTuple]
    ) -> Tuple[List[int], List[int]]:
        """
        Find matching spans between ground truth and predictions.
        
        Each true span can match at most one predicted span and vice versa.
        Matching is done greedily in order of appearance.
        
        Args:
            y_true: Ground truth spans
            y_pred: Predicted spans
            
        Returns:
            (matched_true_indices, matched_pred_indices)
        """
        matched_true = []
        matched_pred = []
        used_pred = set()
        
        for true_idx, true_span in enumerate(y_true):
            for pred_idx, pred_span in enumerate(y_pred):
                if pred_idx not in used_pred:
                    if StrictSpanMatcher.spans_equal(true_span, pred_span):
                        matched_true.append(true_idx)
                        matched_pred.append(pred_idx)
                        used_pred.add(pred_idx)
                        break
        
        return matched_true, matched_pred


def precision_recall_f1(
    y_true: List[List[SpanTuple]],
    y_pred: List[List[SpanTuple]],
    zero_division: str = "warn"
) -> MetricsResult:
    """
    Compute precision, recall, and F1 score with strict span matching.
    
    Metric:
    - Strict Span Match: (start, end, category) must be exactly equal
    - Micro-averaged: aggregate TP, FP, FN across all documents
    
    Args:
        y_true: List of ground truth span lists for each document
        y_pred: List of predicted span lists for each document
        zero_division: How to handle division by zero
                      'warn' = warn and return 0
                      'error' = raise error
                      0 or 1 = return this value
        
    Returns:
        MetricsResult with precision, recall, f1
        
    Raises:
        ValueError: If inputs have mismatched lengths or invalid format
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Mismatched lengths: y_true has {len(y_true)} docs, "
            f"y_pred has {len(y_pred)} docs"
        )
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for doc_idx, (true_spans, pred_spans) in enumerate(zip(y_true, y_pred)):
        # Validate format
        if not isinstance(true_spans, list):
            raise ValueError(
                f"Doc {doc_idx}: y_true[{doc_idx}] is {type(true_spans).__name__}, "
                f"expected list"
            )
        if not isinstance(pred_spans, list):
            raise ValueError(
                f"Doc {doc_idx}: y_pred[{doc_idx}] is {type(pred_spans).__name__}, "
                f"expected list"
            )
        
        # Find matches
        matched_true, matched_pred = StrictSpanMatcher.find_matches(true_spans, pred_spans)
        
        # Count metrics
        tp = len(matched_true)
        fp = len(pred_spans) - tp
        fn = len(true_spans) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Compute metrics
    precision = _compute_precision(total_tp, total_fp, zero_division)
    recall = _compute_recall(total_tp, total_fn, zero_division)
    f1 = _compute_f1(precision, recall, zero_division)
    
    return MetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        support=total_tp + total_fn
    )


def per_category_metrics(
    y_true: List[List[SpanTuple]],
    y_pred: List[List[SpanTuple]],
    categories: Optional[List[str]] = None,
    zero_division: str = "warn"
) -> Dict[str, MetricsResult]:
    """
    Compute metrics for each entity category separately.
    
    Args:
        y_true: Ground truth spans
        y_pred: Predicted spans
        categories: List of valid categories (if None, extracted from y_true and y_pred)
        zero_division: How to handle division by zero
        
    Returns:
        Dictionary mapping category to MetricsResult
    """
    if categories is None:
        categories = _extract_categories(y_true, y_pred)
    
    results = {}
    
    for category in categories:
        # Filter spans by category
        y_true_cat = [
            [span for span in doc_spans if span[2] == category]
            for doc_spans in y_true
        ]
        y_pred_cat = [
            [span for span in doc_spans if span[2] == category]
            for doc_spans in y_pred
        ]
        
        # Compute metrics
        metrics = precision_recall_f1(y_true_cat, y_pred_cat, zero_division=zero_division)
        results[category] = metrics
    
    return results


def micro_f1_score(
    y_true: List[List[SpanTuple]],
    y_pred: List[List[SpanTuple]],
    zero_division: str = "warn"
) -> float:
    """
    Compute micro-averaged F1 score.
    
    Args:
        y_true: Ground truth spans
        y_pred: Predicted spans
        zero_division: How to handle division by zero
        
    Returns:
        F1 score (float between 0 and 1)
    """
    metrics = precision_recall_f1(y_true, y_pred, zero_division=zero_division)
    return metrics.f1


def _compute_precision(tp: int, fp: int, zero_division: str = "warn") -> float:
    """Compute precision from TP and FP."""
    denominator = tp + fp
    
    if denominator == 0:
        return _handle_zero_division(zero_division, "precision")
    
    return tp / denominator


def _compute_recall(tp: int, fn: int, zero_division: str = "warn") -> float:
    """Compute recall from TP and FN."""
    denominator = tp + fn
    
    if denominator == 0:
        return _handle_zero_division(zero_division, "recall")
    
    return tp / denominator


def _compute_f1(precision: float, recall: float, zero_division: str = "warn") -> float:
    """Compute F1 from precision and recall."""
    denominator = precision + recall
    
    if denominator == 0:
        return _handle_zero_division(zero_division, "f1")
    
    return 2 * (precision * recall) / denominator


def _handle_zero_division(zero_division: str, metric_name: str) -> float:
    """Handle zero division according to policy."""
    if zero_division == "warn":
        logger.warning(f"{metric_name}: zero division, returning 0.0")
        return 0.0
    elif zero_division == "error":
        raise ValueError(f"Zero division in {metric_name} computation")
    else:
        # Assume zero_division is a number (0 or 1)
        return float(zero_division)


def _extract_categories(
    y_true: List[List[SpanTuple]],
    y_pred: List[List[SpanTuple]]
) -> List[str]:
    """Extract unique categories from predictions and ground truth."""
    categories = set()
    
    for doc_spans in y_true:
        for span in doc_spans:
            categories.add(span[2])
    
    for doc_spans in y_pred:
        for span in doc_spans:
            categories.add(span[2])
    
    return sorted(list(categories))


def evaluate_dataframe(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    text_col: str = "text",
    spans_col: str = "predictions",
    id_col: str = "id"
) -> Dict[str, Any]:
    """
    Evaluate predictions from dataframes.
    
    Expected dataframe format:
    - id: Document ID
    - text: Original text
    - predictions: List[Tuple[int, int, str]]
    
    Args:
        df_true: Ground truth dataframe
        df_pred: Predictions dataframe
        text_col: Column name for text
        spans_col: Column name for spans/predictions
        id_col: Column name for document ID
        
    Returns:
        Dictionary with overall and per-category metrics
    """
    # Validate dataframes
    if len(df_true) != len(df_pred):
        raise ValueError(
            f"Mismatched dataframe lengths: true has {len(df_true)}, "
            f"pred has {len(df_pred)}"
        )
    
    if id_col not in df_true.columns or id_col not in df_pred.columns:
        raise ValueError(f"Missing '{id_col}' column")
    
    if spans_col not in df_true.columns or spans_col not in df_pred.columns:
        raise ValueError(f"Missing '{spans_col}' column")
    
    # Extract spans
    y_true = []
    y_pred = []
    
    for idx_true, idx_pred in zip(df_true.index, df_pred.index):
        row_true = df_true.loc[idx_true]
        row_pred = df_pred.loc[idx_pred]
        
        # Parse spans
        true_spans = _parse_spans(row_true[spans_col])
        pred_spans = _parse_spans(row_pred[spans_col])
        
        y_true.append(true_spans)
        y_pred.append(pred_spans)
    
    # Compute metrics
    overall_metrics = precision_recall_f1(y_true, y_pred)
    category_metrics = per_category_metrics(y_true, y_pred)
    
    return {
        "overall": overall_metrics,
        "per_category": category_metrics,
        "num_documents": len(y_true),
        "num_true_entities": sum(len(spans) for spans in y_true),
        "num_pred_entities": sum(len(spans) for spans in y_pred)
    }


def _parse_spans(spans_data: Any) -> List[SpanTuple]:
    """Parse spans from various formats."""
    if isinstance(spans_data, list):
        # Already a list
        spans = []
        for span in spans_data:
            if isinstance(span, (tuple, list)) and len(span) == 3:
                spans.append(tuple(span))
            else:
                logger.warning(f"Invalid span format: {span}")
        return spans
    elif isinstance(spans_data, str):
        # Try to parse JSON
        import json
        try:
            data = json.loads(spans_data)
            spans = []
            for span in data:
                if isinstance(span, (tuple, list)) and len(span) == 3:
                    spans.append(tuple(span))
            return spans
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse spans from string: {spans_data}")
            return []
    else:
        return []


def format_metrics_report(
    metrics: MetricsResult,
    per_category: Optional[Dict[str, MetricsResult]] = None,
    digits: int = 4
) -> str:
    """
    Format metrics into a readable report string.
    
    Args:
        metrics: Overall MetricsResult
        per_category: Per-category MetricsResult dictionary
        digits: Number of decimal places
        
    Returns:
        Formatted report string
    """
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append("STRICT SPAN + CATEGORY F1 METRICS")
    lines.append("=" * 70)
    
    # Overall metrics
    lines.append("\nOVERALL METRICS:")
    lines.append(f"  Precision:  {metrics.precision:.{digits}f}")
    lines.append(f"  Recall:     {metrics.recall:.{digits}f}")
    lines.append(f"  F1-Score:   {metrics.f1:.{digits}f}")
    lines.append(f"  Support:    {metrics.support}")
    lines.append(f"  TP:         {metrics.tp}")
    lines.append(f"  FP:         {metrics.fp}")
    lines.append(f"  FN:         {metrics.fn}")
    
    # Per-category metrics
    if per_category:
        lines.append("\nPER-CATEGORY METRICS:")
        lines.append("-" * 70)
        
        # Header row
        lines.append(
            f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<8}"
        )
        lines.append("-" * 70)
        
        # Data rows
        for category in sorted(per_category.keys()):
            cat_metrics = per_category[category]
            lines.append(
                f"{category:<20} "
                f"{cat_metrics.precision:<12.{digits}f} "
                f"{cat_metrics.recall:<12.{digits}f} "
                f"{cat_metrics.f1:<12.{digits}f} "
                f"{cat_metrics.support:<8}"
            )
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
