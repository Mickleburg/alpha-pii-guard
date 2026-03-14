"""Evaluation script for strict span F1 metric."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import pandas as pd

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


class StrictSpanF1Evaluator:
    """Compute strict span F1 metric."""
    
    @staticmethod
    def parse_prediction(pred_str: str) -> List[SpanTuple]:
        """
        Parse prediction string to spans.
        
        Args:
            pred_str: JSON string or list representation
            
        Returns:
            List of (start, end, category) tuples
        """
        try:
            if isinstance(pred_str, str):
                spans = json.loads(pred_str)
            else:
                spans = pred_str
            
            # Validate format
            result = []
            for span in spans:
                if isinstance(span, (list, tuple)) and len(span) == 3:
                    start, end, category = span
                    result.append((int(start), int(end), str(category)))
            
            return result
        except Exception as e:
            logger.warning(f"Failed to parse prediction: {e}")
            return []
    
    @staticmethod
    def spans_match(span1: SpanTuple, span2: SpanTuple) -> bool:
        """Check if two spans match exactly."""
        start1, end1, cat1 = span1
        start2, end2, cat2 = span2
        
        return start1 == start2 and end1 == end2 and cat1 == cat2
    
    @staticmethod
    def compute_metrics(
        gold_spans: List[SpanTuple],
        pred_spans: List[SpanTuple]
    ) -> Dict[str, float]:
        """
        Compute strict span F1 metrics.
        
        Args:
            gold_spans: Gold standard spans
            pred_spans: Predicted spans
            
        Returns:
            Dictionary with precision, recall, f1
        """
        if not pred_spans and not gold_spans:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        
        if not gold_spans:
            # No gold spans to find
            if pred_spans:
                return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(pred_spans), "fn": 0}
            else:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        
        if not pred_spans:
            # No predictions but gold spans exist
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gold_spans)}
        
        # Convert to sets for matching
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        
        # True positives: predictions that match gold
        tp = len(gold_set & pred_set)
        
        # False positives: predictions that don't match any gold
        fp = len(pred_set - gold_set)
        
        # False negatives: gold spans not predicted
        fn = len(gold_set - pred_set)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    @staticmethod
    def compute_category_metrics(
        gold_spans: List[SpanTuple],
        pred_spans: List[SpanTuple]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per category.
        
        Args:
            gold_spans: Gold standard spans
            pred_spans: Predicted spans
            
        Returns:
            Dictionary mapping category -> metrics
        """
        # Group by category
        gold_by_cat = defaultdict(list)
        pred_by_cat = defaultdict(list)
        
        for start, end, cat in gold_spans:
            gold_by_cat[cat].append((start, end, cat))
        
        for start, end, cat in pred_spans:
            pred_by_cat[cat].append((start, end, cat))
        
        # All categories
        all_cats = set(gold_by_cat.keys()) | set(pred_by_cat.keys())
        
        # Compute per category
        results = {}
        for cat in sorted(all_cats):
            gold_cat = gold_by_cat.get(cat, [])
            pred_cat = pred_by_cat.get(cat, [])
            
            metrics = StrictSpanF1Evaluator.compute_metrics(gold_cat, pred_cat)
            results[cat] = metrics
        
        return results


def load_gold_predictions(path: str) -> Dict[str, List[SpanTuple]]:
    """Load gold predictions from file."""
    gold_preds = {}
    
    df = pd.read_csv(path)
    
    for _, row in df.iterrows():
        doc_id = str(row["id"])
        pred_str = row["prediction"]
        spans = StrictSpanF1Evaluator.parse_prediction(pred_str)
        gold_preds[doc_id] = spans
    
    return gold_preds


def load_pred_predictions(path: str) -> Dict[str, List[SpanTuple]]:
    """Load predicted predictions from file."""
    pred_preds = {}
    
    df = pd.read_csv(path)
    
    for _, row in df.iterrows():
        doc_id = str(row["id"])
        pred_str = row["prediction"]
        spans = StrictSpanF1Evaluator.parse_prediction(pred_str)
        pred_preds[doc_id] = spans
    
    return pred_preds


def evaluate(
    gold_path: str,
    pred_path: str,
    output_path: str = None
) -> Dict[str, any]:
    """
    Evaluate predictions against gold standard.
    
    Args:
        gold_path: Path to gold predictions CSV
        pred_path: Path to predicted predictions CSV
        output_path: Optional output path for results
        
    Returns:
        Evaluation results
    """
    logger.info(f"Loading gold predictions from {gold_path}...")
    gold_preds = load_gold_predictions(gold_path)
    logger.info(f"Loaded {len(gold_preds)} gold documents")
    
    logger.info(f"Loading predicted predictions from {pred_path}...")
    pred_preds = load_pred_predictions(pred_path)
    logger.info(f"Loaded {len(pred_preds)} predicted documents")
    
    # Find common documents
    common_ids = set(gold_preds.keys()) & set(pred_preds.keys())
    logger.info(f"Common documents: {len(common_ids)}")
    
    if not common_ids:
        logger.error("No common documents found!")
        return {}
    
    # Compute micro metrics (all spans combined)
    all_gold = []
    all_pred = []
    
    for doc_id in common_ids:
        all_gold.extend(gold_preds[doc_id])
        all_pred.extend(pred_preds[doc_id])
    
    micro_metrics = StrictSpanF1Evaluator.compute_metrics(all_gold, all_pred)
    
    # Compute per-category metrics
    category_metrics = StrictSpanF1Evaluator.compute_category_metrics(all_gold, all_pred)
    
    # Compute macro metrics (average across categories)
    if category_metrics:
        macro_precision = sum(m["precision"] for m in category_metrics.values()) / len(category_metrics)
        macro_recall = sum(m["recall"] for m in category_metrics.values()) / len(category_metrics)
        macro_f1 = sum(m["f1"] for m in category_metrics.values()) / len(category_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    # Build results
    results = {
        "micro": micro_metrics,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "per_category": dict(category_metrics),
        "num_documents": len(common_ids),
        "num_gold_spans": len(all_gold),
        "num_pred_spans": len(all_pred)
    }
    
    # Save results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_path}")
    
    return results


def print_results(results: Dict[str, any]) -> None:
    """Print evaluation results."""
    if not results:
        logger.error("No results to print")
        return
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Summary stats
    print(f"\nDocuments: {results['num_documents']}")
    print(f"Gold spans: {results['num_gold_spans']}")
    print(f"Predicted spans: {results['num_pred_spans']}")
    
    # Micro metrics
    micro = results["micro"]
    print("\n" + "-"*80)
    print("MICRO METRICS (all spans combined):")
    print("-"*80)
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall:    {micro['recall']:.4f}")
    print(f"  F1 Score:  {micro['f1']:.4f}")
    print(f"  True Positives:  {micro['tp']}")
    print(f"  False Positives: {micro['fp']}")
    print(f"  False Negatives: {micro['fn']}")
    
    # Macro metrics
    macro = results["macro"]
    print("\n" + "-"*80)
    print("MACRO METRICS (average across categories):")
    print("-"*80)
    print(f"  Precision: {macro['precision']:.4f}")
    print(f"  Recall:    {macro['recall']:.4f}")
    print(f"  F1 Score:  {macro['f1']:.4f}")
    
    # Per-category metrics
    if results["per_category"]:
        print("\n" + "-"*80)
        print("PER-CATEGORY METRICS:")
        print("-"*80)
        
        # Header
        print(f"\n{'Category':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP/FP/FN':<15}")
        print("-" * 80)
        
        for category in sorted(results["per_category"].keys()):
            metrics = results["per_category"][category]
            print(
                f"{category:<25} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} "
                f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']:<12}"
            )
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate strict span F1")
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold predictions CSV"
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to predicted predictions CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Evaluate
    results = evaluate(args.gold, args.pred, args.output)
    
    # Print
    print_results(results)


if __name__ == "__main__":
    main()
