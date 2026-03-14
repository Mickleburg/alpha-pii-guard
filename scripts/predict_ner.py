"""Script to run NER inference on test data."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.ner.inference import NERInference
from src.data.readers import load_test_dataset

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


def postprocess_spans(spans: List[SpanTuple], text: str) -> List[SpanTuple]:
    """
    Post-process spans to ensure quality.
    
    Args:
        spans: Raw spans from model
        text: Original text
        
    Returns:
        Cleaned spans
    """
    if not spans:
        return []
    
    # Remove duplicates
    unique_spans = {}
    for start, end, category in spans:
        key = (start, end, category)
        unique_spans[key] = (start, end, category)
    
    result = list(unique_spans.values())
    
    # Sort by start
    result.sort(key=lambda x: x[0])
    
    # Final validation
    text_len = len(text)
    valid = []
    for start, end, category in result:
        if 0 <= start < end <= text_len:
            span_text = text[start:end]
            if span_text.strip():
                valid.append((start, end, category))
    
    return valid


def run_inference(
    model_path: str,
    test_data_path: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 32
) -> None:
    """
    Run inference on test data.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test CSV
        output_path: Output path for predictions
        device: Device to use
        batch_size: Batch size for processing
    """
    logger.info("Loading model...")
    
    # Load model
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    model = NERInference(model_path, device=device)
    
    logger.info("Loading test data...")
    
    # Load test data
    test_docs = load_test_dataset(test_data_path)
    
    logger.info(f"Processing {len(test_docs)} documents...")
    
    # Run inference
    predictions = []
    
    for doc_idx, doc in enumerate(test_docs):
        if (doc_idx + 1) % 100 == 0:
            logger.info(f"Processed {doc_idx + 1}/{len(test_docs)}")
        
        text = doc["text"]
        doc_id = doc["id"]
        
        # Predict
        spans = model.predict(text)
        
        # Post-process
        spans = postprocess_spans(spans, text)
        
        predictions.append({
            "id": doc_id,
            "text": text,
            "predictions": spans
        })
    
    logger.info(f"Saving predictions to {output_path}...")
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            line = json.dumps(pred, ensure_ascii=False)
            f.write(line + "\n")
    
    logger.info(f"Saved {len(predictions)} predictions")
    
    # Also save as CSV
    csv_output = output_path.with_suffix(".csv")
    df_data = []
    for pred in predictions:
        df_data.append({
            "id": pred["id"],
            "text": pred["text"],
            "predictions": json.dumps(pred["predictions"])
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_output, index=False)
    logger.info(f"Also saved as CSV: {csv_output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run NER inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ner_checkpoint/model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/raw/private_test_dataset.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/answer/predictions_ner.jsonl",
        help="Output path for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("NER INFERENCE")
    logger.info("="*60)
    
    # Run inference
    run_inference(
        args.model_path,
        args.test_data,
        args.output,
        device=args.device,
        batch_size=args.batch_size
    )
    
    logger.info("="*60)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
