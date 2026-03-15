import argparse
import pandas as pd
import ast
import os
from pathlib import Path

from src.regex_detector import detect_pii
from src.ner_model import NERModel
from src.merge_predictions import merge_predictions
from src.prepare_data import read_train_dataset, read_test_dataset
from src.evaluate import compute_metrics
from src.utils import ensure_dirs

ensure_dirs()

def prepare_command(args):
    """Копируем данные в папки (нет валидации на отдельной)."""
    print("Preparing data...")
    
    # Копируем train и test
    if os.path.exists("data/raw/train_dataset.tsv"):
        train_df = read_train_dataset("data/raw/train_dataset.tsv")
        train_df.to_csv("data/processed/train.csv", index=False)
        print(f"Train: {len(train_df)} samples")
    
    if os.path.exists("data/raw/private_test_dataset.csv"):
        test_df = read_test_dataset("data/raw/private_test_dataset.csv")
        test_df.to_csv("data/processed/test.csv", index=False)
        print(f"Test: {len(test_df)} samples")

def regex_command(args):
    """Запуск regex детектора на test датасете."""
    input_file = args.input or "data/processed/test.csv"
    output_file = args.output or "data/answer/regex_predictions.csv"
    
    print(f"Running regex detector on {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    if input_file.endswith(".tsv"):
        df = pd.read_csv(input_file, sep="\t", dtype=str)
    else:
        df = pd.read_csv(input_file, dtype=str)
    
    df["text"] = df["text"].astype(str)
    predictions = []
    
    for text in df["text"]:
        pred = detect_pii(text)
        # Преобразуем список кортежей в строку для сохранения
        predictions.append(pred)
    
    result_df = pd.DataFrame({
        "prediction": predictions,
    })
    
    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])
    
    result_df["text"] = df["text"]
    
    result_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print(f"  Sample prediction: {predictions[0] if predictions else '[]'}")
    
    # Если есть target - показываем метрики
    if "target" in df.columns:
        targets = []
        for t in df["target"]:
            try:
                if pd.isna(t) or t == "[]":
                    targets.append([])
                else:
                    targets.append(ast.literal_eval(str(t)))
            except:
                targets.append([])
        
        metrics = compute_metrics(predictions, targets)
        print(f"  Regex metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['micro_f1']:.4f}")

def ner_train_command(args):
    """Обучение NER модели на всём train датасете."""
    print("Training NER model...")
    
    if not os.path.exists("data/processed/train.csv"):
        print("❌ Run 'python main.py prepare' first!")
        return
    
    train_df = pd.read_csv("data/processed/train.csv", dtype=str)
    
    # Парсим target
    def parse_target(t):
        try:
            if pd.isna(t) or t == "[]":
                return []
            return ast.literal_eval(str(t))
        except:
            return []
    
    train_df["target"] = train_df["target"].apply(parse_target)
    
    print(f"Training on {len(train_df)} samples...")
    model = NERModel()
    model.train(train_df, epochs=3, batch_size=8, max_len=512)
    print("✓ NER model trained and saved")

def ner_predict_command(args):
    """Предсказание NER на test датасете."""
    input_file = args.input or "data/processed/test.csv"
    output_file = args.output or "data/answer/ner_predictions.csv"
    
    print(f"Running NER on {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file, dtype=str)
    df["text"] = df["text"].astype(str)
    
    model = NERModel()
    model.load()
    
    print(f"Predicting on {len(df)} texts...")
    predictions = model.predict_batch(df["text"].tolist())
    
    result_df = pd.DataFrame({
        "prediction": predictions,
    })
    
    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])
    
    result_df["text"] = df["text"]
    
    result_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print(f"  Sample prediction: {predictions[0] if predictions else '[]'}")
    
    # Если есть target - показываем метрики
    if "target" in df.columns:
        targets = []
        for t in df["target"]:
            try:
                if pd.isna(t) or t == "[]":
                    targets.append([])
                else:
                    targets.append(ast.literal_eval(str(t)))
            except:
                targets.append([])
        
        metrics = compute_metrics(predictions, targets)
        print(f"  NER metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['micro_f1']:.4f}")

def merge_command(args):
    """Merge regex и NER predictions."""
    print("Merging regex and NER predictions...")
    
    regex_file = args.regex or "data/answer/regex_predictions.csv"
    ner_file = args.ner or "data/answer/ner_predictions.csv"
    output_file = args.output or "data/answer/merged_predictions.csv"
    
    if not os.path.exists(regex_file):
        print(f"❌ File not found: {regex_file}")
        return
    
    if not os.path.exists(ner_file):
        print(f"❌ File not found: {ner_file}")
        return
    
    regex_df = pd.read_csv(regex_file)
    ner_df = pd.read_csv(ner_file)
    
    merged = []
    
    for i, (regex_pred, ner_pred) in enumerate(zip(regex_df["prediction"], ner_df["prediction"])):
        # Парсим predictions
        try:
            if isinstance(regex_pred, str):
                regex_spans = ast.literal_eval(regex_pred) if regex_pred and regex_pred != "[]" else []
            else:
                regex_spans = regex_pred if regex_pred else []
        except:
            regex_spans = []
        
        try:
            if isinstance(ner_pred, str):
                ner_spans = ast.literal_eval(ner_pred) if ner_pred and ner_pred != "[]" else []
            else:
                ner_spans = ner_pred if ner_pred else []
        except:
            ner_spans = []
        
        merged_spans = merge_predictions(regex_spans, ner_spans)
        merged.append(merged_spans)
    
    result_df = pd.DataFrame({
        "prediction": merged,
    })
    
    if "id" in regex_df.columns:
        result_df.insert(0, "id", regex_df["id"])
    
    result_df["text"] = regex_df["text"]
    
    result_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print(f"  Sample merged: {merged[0] if merged else '[]'}")

def all_command(args):
    """Полный pipeline: подготовка -> обучение -> предсказания -> merge."""
    print("=" * 60)
    print("Running full NER pipeline")
    print("=" * 60)
    
    print("\n[1/5] Preparing data...")
    prepare_command(args)
    
    print("\n[2/5] Training NER model...")
    ner_train_command(args)
    
    print("\n[3/5] Running regex detector...")
    regex_command({**vars(args), "input": "data/processed/test.csv", "output": "data/answer/regex_predictions.csv"})
    
    print("\n[4/5] Running NER predictions...")
    ner_predict_command({**vars(args), "input": "data/processed/test.csv", "output": "data/answer/ner_predictions.csv"})
    
    print("\n[5/5] Merging predictions...")
    merge_command({**vars(args), "regex": "data/answer/regex_predictions.csv", "ner": "data/answer/ner_predictions.csv", "output": "data/answer/merged_predictions.csv"})
    
    print("\n" + "=" * 60)
    print("✓ Pipeline completed!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Pipeline for Russian PII detection")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Prepare
    subparsers.add_parser("prepare", help="Prepare data")
    
    # Regex
    regex_parser = subparsers.add_parser("regex", help="Run regex detector")
    regex_parser.add_argument("--input", default=None, help="Input file path")
    regex_parser.add_argument("--output", default=None, help="Output file path")
    
    # NER train
    subparsers.add_parser("ner_train", help="Train NER model")
    
    # NER predict
    ner_pred_parser = subparsers.add_parser("ner_predict", help="Run NER prediction")
    ner_pred_parser.add_argument("--input", default=None, help="Input file path")
    ner_pred_parser.add_argument("--output", default=None, help="Output file path")
    
    # Merge
    merge_parser = subparsers.add_parser("merge", help="Merge regex and NER predictions")
    merge_parser.add_argument("--regex", default=None, help="Regex predictions file")
    merge_parser.add_argument("--ner", default=None, help="NER predictions file")
    merge_parser.add_argument("--output", default=None, help="Output file path")
    
    # All
    subparsers.add_parser("all", help="Run full pipeline")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "regex":
        regex_command(args)
    elif args.command == "ner_train":
        ner_train_command(args)
    elif args.command == "ner_predict":
        ner_predict_command(args)
    elif args.command == "merge":
        merge_command(args)
    elif args.command == "all":
        all_command(args)
    else:
        parser.print_help()