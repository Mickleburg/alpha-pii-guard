"""Script to run merged (regex + NER) inference on test data."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.regex.detector import create_detector
from src.ner.inference import NERInference
from src.merge.pipeline import PIIDetectionPipeline
from src.data.readers import load_test_dataset

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


def format_predictions(spans: List[SpanTuple]) -> str:
    """Format spans as string for CSV."""
    return json.dumps(spans, ensure_ascii=False)


def run_merged_prediction(
    test_data_path: str,
    output_path: str,
    model_path: str,
    use_context_rules: bool = True,
    merge_strategy: str = "regex_priority",
    device: str = "cuda"
) -> None:
    """
    Run merged prediction on test data.
    
    Args:
        test_data_path: Path to test CSV
        output_path: Output path for predictions
        model_path: Path to trained NER model
        use_context_rules: Whether to use regex context rules
        merge_strategy: Strategy for merging (regex_priority, ner_priority, union)
        device: Device to use
    """
    logger.info("Loading components...")
    
    # Create regex detector
    regex_detector = create_detector(use_context_rules=use_context_rules)
    
    # Load NER model
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    try:
        ner_detector = NERInference(model_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load NER model: {e}")
        return
    
    # Create pipeline
    pipeline = PIIDetectionPipeline(
        regex_detector=regex_detector,
        ner_detector=ner_detector,
        merge_strategy=merge_strategy
    )
    
    logger.info("Loading test data...")
    
    # Load test data
    test_docs = load_test_dataset(test_data_path)
    
    logger.info(f"Processing {len(test_docs)} documents...")
    
    # Run predictions
    predictions = []
    
    for doc_idx, doc in enumerate(test_docs):
        if (doc_idx + 1) % 50 == 0:
            logger.info(f"Processed {doc_idx + 1}/{len(test_docs)}")
        
        text = doc["text"]
        doc_id = doc["id"]
        
        # Predict
        spans = pipeline.predict(text)
        
        predictions.append({
            "id": doc_id,
            "prediction": format_predictions(spans)
        })
    
    logger.info(f"Saving predictions to {output_path}...")
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    
    # Print statistics
    total_spans = sum(len(json.loads(p["prediction"])) for p in predictions)
    logger.info(f"Total PII spans detected: {total_spans}")
    logger.info(f"Average spans per document: {total_spans / len(predictions):.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run merged PII detection (regex + NER)")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/raw/private_test_dataset.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/answer/final_predictions.csv",
        help="Output path for predictions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ner_checkpoint/model",
        help="Path to trained NER model"
    )
    parser.add_argument(
        "--merge-strategy",
        type=str,
        default="regex_priority",
        choices=["regex_priority", "ner_priority", "union"],
        help="Strategy for merging predictions"
    )
    parser.add_argument(
        "--use-context",
        action="store_true",
        default=True,
        help="Use context rules in regex"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("MERGED PII DETECTION (REGEX + NER)")
    logger.info("="*60)
    logger.info(f"Merge strategy: {args.merge_strategy}")
    
    # Run prediction
    run_merged_prediction(
        args.test_data,
        args.output,
        args.model_path,
        use_context_rules=args.use_context,
        merge_strategy=args.merge_strategy,
        device=args.device
    )
    
    logger.info("="*60)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
