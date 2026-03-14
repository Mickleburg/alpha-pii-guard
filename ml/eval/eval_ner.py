# File: ml/eval/eval_ner.py
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.pipelines.infer_ner import NERModel

def load_test_data(test_path: Path) -> List[Dict]:
    """Load test dataset from JSONL file."""
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_strict_match_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    ground_truths: List[List[Tuple[int, int, str]]]
) -> Dict:
    """Compute strict span+category match metrics.
    
    TP: exact (start, end, category) match
    FP: predicted but not in ground truth
    FN: in ground truth but not predicted
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Per-category metrics
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred_spans, true_spans in zip(predictions, ground_truths):
        # Convert to sets for matching
        pred_set = set(pred_spans)
        true_set = set(true_spans)
        
        # True positives: predicted and in ground truth
        tp_spans = pred_set & true_set
        total_tp += len(tp_spans)
        
        # False positives: predicted but not in ground truth
        fp_spans = pred_set - true_set
        total_fp += len(fp_spans)
        
        # False negatives: in ground truth but not predicted
        fn_spans = true_set - pred_set
        total_fn += len(fn_spans)
        
        # Per-category stats
        for span in tp_spans:
            category = span[2]
            category_stats[category]['tp'] += 1
        
        for span in fp_spans:
            category = span[2]
            category_stats[category]['fp'] += 1
        
        for span in fn_spans:
            category = span[2]
            category_stats[category]['fn'] += 1
    
    # Micro-averaged metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Per-category metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
        
        category_metrics[category] = {
            'precision': cat_precision,
            'recall': cat_recall,
            'f1': cat_f1,
            'support': tp + fn
        }
    
    return {
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'category_metrics': category_metrics
    }

def main():
    # Paths
    test_data_path = Path('data/processed/test_raw.jsonl')
    model_dir = Path('ml/models/ner')
    output_path = Path('docs/eval_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading test data...")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    print("\nLoading NER model...")
    ner_model = NERModel(model_dir=str(model_dir))
    
    # Run predictions
    print("\nRunning predictions...")
    texts = [item['text'] for item in test_data]
    ground_truths = [item['entities'] for item in test_data]
    
    # Batch prediction for efficiency
    predictions = ner_model.predict_batch(texts)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_strict_match_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Strict Span + Category Match)")
    print("="*60)
    print(f"\nMicro-averaged metrics:")
    print(f"  Precision: {results['micro_precision']:.4f}")
    print(f"  Recall:    {results['micro_recall']:.4f}")
    print(f"  F1 Score:  {results['micro_f1']:.4f}")
    print(f"\nCounts:")
    print(f"  True Positives:  {results['total_tp']}")
    print(f"  False Positives: {results['total_fp']}")
    print(f"  False Negatives: {results['total_fn']}")
    
    print("\n" + "-"*60)
    print("Per-category metrics:")
    print("-"*60)
    print(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*60)
    
    for category, metrics in sorted(results['category_metrics'].items(), key=lambda x: -x[1]['f1']):
        print(f"{category:<30} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")
    
    print("="*60)
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
