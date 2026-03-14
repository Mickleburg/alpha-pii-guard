# File: scripts/run_inference_on_tsv.py
"""
Run NER inference on TSV/CSV dataset and save predictions.

Reads a tabular file with at least 'text' column, runs batch prediction, and saves:
1) JSONL with {"text": ..., "predictions": [[start, end, category], ...]}
2) TSV with original columns + predictions column formatted as:
   [(start, end, "CATEGORY"), ...]
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipelines.infer_ner import NERModel


Prediction = Tuple[int, int, str]


def load_table(input_path: Path, sep: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path, sep=sep)
    except Exception as exc:
        raise ValueError(f"Failed to read file '{input_path}': {exc}") from exc

    if "text" not in df.columns:
        raise ValueError(f"Input file must contain 'text' column. Found: {list(df.columns)}")

    return df


def prepare_texts(df: pd.DataFrame) -> List[str]:
    texts: List[str] = []

    for _, row in df.iterrows():
        text = row["text"]
        if pd.isna(text):
            texts.append("")
        elif isinstance(text, str):
            texts.append(text)
        else:
            texts.append(str(text))

    return texts


def format_predictions_as_tuple_string(predictions: List[Prediction]) -> str:
    if not predictions:
        return "[]"

    parts = []
    for start, end, category in predictions:
        safe_category = category.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'({start}, {end}, "{safe_category}")')
    return "[" + ", ".join(parts) + "]"


def save_predictions_jsonl(
    texts: List[str],
    predictions: List[List[Prediction]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text, pred in zip(texts, predictions):
            pred_list = [[start, end, category] for start, end, category in pred]
            entry = {
                "text": text,
                "predictions": pred_list,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_predictions_tsv(
    df: pd.DataFrame,
    predictions: List[List[Prediction]],
    output_tsv_path: Path,
) -> None:
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = df.copy()
    result_df["predictions"] = [
        format_predictions_as_tuple_string(pred) for pred in predictions
    ]
    result_df.to_csv(output_tsv_path, sep="\t", index=False, encoding="utf-8")


def resolve_output_tsv_path(output_jsonl_path: Path, explicit_output_tsv: str | None) -> Path:
    if explicit_output_tsv:
        return Path(explicit_output_tsv)

    if output_jsonl_path.suffix.lower() == ".jsonl":
        return output_jsonl_path.with_suffix(".tsv")

    return output_jsonl_path.parent / f"{output_jsonl_path.stem}.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER inference on tabular dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/private/test_dataset.csv",
        help="Input file path",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help=r"Column separator in input file. Use '\t' for TSV or ',' for CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/private_test_predictions.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default="data/processed/private_test_predictions.tsv",
        help='Output TSV file path with predictions as [(start, end, "CATEGORY"), ...]',
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ml/models/ner",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl_path = Path(args.output)
    output_tsv_path = resolve_output_tsv_path(output_jsonl_path, args.output_tsv)
    model_dir = Path(args.model_dir)

    print("=" * 70)
    print("NER INFERENCE ON TABLE")
    print("=" * 70)
    print(f"Input:       {input_path}")
    print(f"Separator:   {repr(args.sep)}")
    print(f"Output JSONL:{output_jsonl_path}")
    print(f"Output TSV:  {output_tsv_path}")
    print(f"Model dir:   {model_dir}")
    print(f"Batch size:  {args.batch_size}")
    print()

    if args.batch_size <= 0:
        print("❌ ERROR: --batch_size must be a positive integer")
        sys.exit(1)

    if not model_dir.exists():
        print(f"❌ ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print("Loading input data...")
    try:
        df = load_table(input_path, args.sep)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ ERROR: Failed to load data: {e}")
        sys.exit(1)

    print("Preparing texts...")
    texts = prepare_texts(df)
    print(f"✓ Prepared {len(texts)} texts")

    print(f"\nLoading NER model from {model_dir}...")
    try:
        model = NERModel(model_dir=str(model_dir))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        sys.exit(1)

    print(f"\nRunning predictions (batch_size={args.batch_size})...")
    all_predictions: List[List[Prediction]] = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch_texts = texts[i:i + args.batch_size]
        try:
            batch_predictions = model.predict_batch(batch_texts)
            normalized_batch: List[List[Prediction]] = []

            for pred in batch_predictions:
                normalized_pred: List[Prediction] = []
                for start, end, category in pred:
                    normalized_pred.append((int(start), int(end), str(category)))
                normalized_batch.append(normalized_pred)

            all_predictions.extend(normalized_batch)
        except Exception as e:
            print(f"\n❌ ERROR: Prediction failed at batch starting from row {i}: {e}")
            sys.exit(1)

    if len(all_predictions) != len(texts):
        print(f"❌ ERROR: Prediction count mismatch: {len(all_predictions)} vs {len(texts)}")
        sys.exit(1)

    print("✓ Predictions completed")

    total_texts = len(texts)
    total_entities = sum(len(pred) for pred in all_predictions)
    non_empty = sum(1 for pred in all_predictions if pred)

    print("\nPrediction statistics:")
    print(f"  Total texts:         {total_texts}")
    print(f"  Texts with entities: {non_empty} ({(non_empty / total_texts * 100):.1f}%)" if total_texts else "  Texts with entities: 0 (0.0%)")
    print(f"  Total entities:      {total_entities}")
    print(f"  Avg per text:        {(total_entities / total_texts):.2f}" if total_texts else "  Avg per text:        0.00")

    print(f"\nSaving JSONL predictions to {output_jsonl_path}...")
    try:
        save_predictions_jsonl(texts, all_predictions, output_jsonl_path)
        print(f"✓ Saved {len(all_predictions)} JSONL predictions")
    except Exception as e:
        print(f"❌ ERROR: Failed to save JSONL predictions: {e}")
        sys.exit(1)

    print(f"Saving TSV predictions to {output_tsv_path}...")
    try:
        save_predictions_tsv(df, all_predictions, output_tsv_path)
        print(f"✓ Saved {len(all_predictions)} TSV predictions")
    except Exception as e:
        print(f"❌ ERROR: Failed to save TSV predictions: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ Inference completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
