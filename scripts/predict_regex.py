"""Script to run regex-only inference on test data."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.regex.detector import create_detector
from src.data.readers import load_test_dataset

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


def format_predictions(spans: List[SpanTuple]) -> str:
    """Format spans as string for CSV."""
    return json.dumps(spans, ensure_ascii=False)


def run_regex_prediction(
    test_data_path: str,
    output_path: str,
    use_context_rules: bool = True
) -> None:
    """
    Run regex-only prediction on test data.
    
    Args:
        test_data_path: Path to test CSV
        output_path: Output path for predictions
        use_context_rules: Whether to use context filtering
    """
    logger.info("Loading regex detector...")
    
    # Create detector
    detector = create_detector(use_context_rules=use_context_rules)
    
    logger.info("Loading test data...")
    
    # Load test data
    test_docs = load_test_dataset(test_data_path)
    
    logger.info(f"Processing {len(test_docs)} documents...")
    
    # Run predictions
    predictions = []
    
    for doc_idx, doc in enumerate(test_docs):
        if (doc_idx + 1) % 100 == 0:
            logger.info(f"Processed {doc_idx + 1}/{len(test_docs)}")
        
        text = doc["text"]
        doc_id = doc["id"]
        
        # Predict
        spans = detector.predict(text)
        
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run regex-only PII detection")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/raw/private_test_dataset.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/answer/predictions_regex.csv",
        help="Output path for predictions"
    )
    parser.add_argument(
        "--use-context",
        action="store_true",
        default=True,
        help="Use context rules"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("REGEX-ONLY PII DETECTION")
    logger.info("="*60)
    
    # Run prediction
    run_regex_prediction(
        args.test_data,
        args.output,
        use_context_rules=args.use_context
    )
    
    logger.info("="*60)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
