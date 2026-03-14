"""
CLI evaluation script for alpha-pii-guard detectors.

Compares regex_only, ner_only, and merged performance on test data.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ml.pipelines.detect_merged import detect_regex, detect_ner, detect_merged
from ml.eval.metrics_rules_ner import (
    compute_micro_metrics,
    compute_per_label_metrics,
    evaluate_detector,
)


def load_test_data(input_path: str) -> Tuple[List[str], List[List[Tuple[int, int, str]]]]:
    """
    Load test data from TSV or JSONL format.
    
    Expected TSV format:
        text\tentity\tentity_texts
        "Sample text"\t[(0, 5, 'Email')]\t["email@example.com"]
    
    Expected JSONL format:
        {"text": "...", "entities": [[0, 5, "Email"], ...]}
    
    Args:
        input_path: Path to test data file
        
    Returns:
        Tuple of (texts, gold_spans)
    """
    path = Path(input_path)
    texts = []
    gold_spans = []
    
    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
                entities = [tuple(e) for e in data['entities']]
                gold_spans.append(entities)
    
    elif path.suffix == '.tsv':
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                text = parts[0]
                entity_str = parts[1]
                
                # Parse entity list: [(start, end, label), ...]
                try:
                    entities = eval(entity_str)
                    texts.append(text)
                    gold_spans.append(entities)
                except Exception:
                    continue
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return texts, gold_spans


def print_metrics_report(results: Dict[str, Any], mode: str) -> None:
    """
    Print formatted metrics report to console.
    
    Args:
        results: Results dict from evaluate_detector
        mode: Detector mode name
    """
    print(f"\n{'='*60}")
    print(f"RESULTS: {mode.upper()}")
    print('='*60)
    
    micro = results['micro']
    print(f"\nMicro-averaged metrics:")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall:    {micro['recall']:.4f}")
    print(f"  F1:        {micro['f1']:.4f}")
    print(f"  TP: {micro['tp']}, FP: {micro['fp']}, FN: {micro['fn']}")
    
    per_label = results['per_label']
    if per_label:
        print(f"\nPer-category metrics:")
        print(f"{'Category':<50} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print('-' * 82)
        
        for label in sorted(per_label.keys()):
            metrics = per_label[label]
            print(f"{label:<50} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dict
        output_path: Path to output JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate regex, NER, and merged detectors"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/test.tsv',
        help='Path to test data (TSV or JSONL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/merge_eval_results.json',
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['regex', 'ner', 'merged', 'all'],
        default='all',
        help='Detector mode to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from: {args.input}")
    texts, gold_spans = load_test_data(args.input)
    print(f"Loaded {len(texts)} examples")
    
    all_results = {}
    
    # Evaluate regex
    if args.mode in ('regex', 'all'):
        print("\nEvaluating regex detector...")
        regex_results = evaluate_detector(detect_regex, texts, gold_spans)
        print_metrics_report(regex_results, 'regex')
        all_results['regex'] = regex_results
    
    # Evaluate NER
    if args.mode in ('ner', 'all'):
        print("\nEvaluating NER detector...")
        ner_results = evaluate_detector(detect_ner, texts, gold_spans)
        print_metrics_report(ner_results, 'ner')
        all_results['ner'] = ner_results
    
    # Evaluate merged
    if args.mode in ('merged', 'all'):
        print("\nEvaluating merged detector...")
        merged_results = evaluate_detector(detect_merged, texts, gold_spans)
        print_metrics_report(merged_results, 'merged')
        all_results['merged'] = merged_results
    
    # Save results
    save_results_json(all_results, args.output)
    print("\n" + "="*60)
    print("Evaluation complete")
    print("="*60)


if __name__ == '__main__':
    main()
