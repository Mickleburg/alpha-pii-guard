# File: scripts/run_inference_on_tsv.py
"""
Run NER inference on TSV dataset and save predictions to JSONL.

Reads TSV with 'text' column, runs batch prediction, and saves results.
Output format: one JSON object per line with text and predictions.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipelines.infer_ner import NERModel

def load_tsv(input_path: Path) -> pd.DataFrame:
    """Load TSV file and validate."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path, sep='\t')
    
    if 'text' not in df.columns:
        raise ValueError(f"TSV must contain 'text' column. Found: {list(df.columns)}")
    
    return df

def prepare_texts(df: pd.DataFrame) -> List[str]:
    """Extract and clean texts from dataframe."""
    texts = []
    for idx, row in df.iterrows():
        text = row['text']
        
        # Handle missing or invalid text
        if pd.isna(text):
            texts.append("")
        elif not isinstance(text, str):
            texts.append(str(text))
        else:
            texts.append(text)
    
    return texts

def save_predictions(
    texts: List[str],
    predictions: List[List[tuple]],
    output_path: Path
):
    """Save predictions to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, pred in zip(texts, predictions):
            # Convert tuples to lists for JSON serialization
            pred_list = [[start, end, category] for start, end, category in pred]
            
            entry = {
                'text': text,
                'predictions': pred_list
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(
        description='Run NER inference on TSV dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/private_test_dataset.tsv',
        help='Input TSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/private_test_predictions.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='ml/models/ner',
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)
    
    print("="*70)
    print("NER INFERENCE ON TSV")
    print("="*70)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Check model exists
    if not model_dir.exists():
        print(f"❌ ERROR: Model directory not found: {model_dir}")
        print("Run training first: python ml/pipelines/train_ner.py")
        sys.exit(1)
    
    # Load data
    print("Loading input data...")
    try:
        df = load_tsv(input_path)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ ERROR: Failed to load data: {e}")
        sys.exit(1)
    
    # Prepare texts
    print("Preparing texts...")
    texts = prepare_texts(df)
    print(f"✓ Prepared {len(texts)} texts")
    
    # Load model
    print(f"\nLoading NER model from {model_dir}...")
    try:
        model = NERModel(model_dir=str(model_dir))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Run batch predictions
    print(f"\nRunning predictions (batch_size={args.batch_size})...")
    
    all_predictions = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch_texts = texts[i:i + args.batch_size]
        
        try:
            batch_predictions = model.predict_batch(batch_texts)
            all_predictions.extend(batch_predictions)
        except Exception as e:
            print(f"\n❌ ERROR: Prediction failed at batch {i}: {e}")
            sys.exit(1)
    
    print(f"✓ Predictions completed")
    
    # Calculate statistics
    total_entities = sum(len(pred) for pred in all_predictions)
    non_empty = sum(1 for pred in all_predictions if len(pred) > 0)
    
    print(f"\nPrediction statistics:")
    print(f"  Total texts:         {len(texts)}")
    print(f"  Texts with entities: {non_empty} ({non_empty/len(texts)*100:.1f}%)")
    print(f"  Total entities:      {total_entities}")
    print(f"  Avg per text:        {total_entities/len(texts):.2f}")
    
    # Save results
    print(f"\nSaving predictions to {output_path}...")
    try:
        save_predictions(texts, all_predictions, output_path)
        print(f"✓ Saved {len(all_predictions)} predictions")
    except Exception as e:
        print(f"❌ ERROR: Failed to save predictions: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ Inference completed successfully")
    print("="*70)

if __name__ == '__main__':
    main()
