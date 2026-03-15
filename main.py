import argparse
import ast
import os
import pandas as pd

from src.prepare_data import read_train_dataset, read_test_dataset, save_processed
from src.regex_detector import detect_pii
from src.ner_model import train_ner, predict_dataframe
from src.merge_predictions import merge_predictions
from src.evaluate import compute_metrics, save_metrics
from src.utils import ensure_dirs


TRAIN_PATH = "data/raw/train_dataset.tsv"
TEST_PATH = "data/raw/private_test_dataset.csv"

PROCESSED_TRAIN = "data/processed/train_full.csv"
PROCESSED_TEST = "data/processed/test.csv"

REGEX_TRAIN_OUT = "data/answer/regex_train_predictions.csv"
REGEX_TEST_OUT = "data/answer/regex_predictions.csv"

NER_TRAIN_OUT = "data/answer/ner_train_predictions.csv"
NER_TEST_OUT = "data/answer/ner_predictions.csv"

MERGED_TRAIN_OUT = "data/answer/merged_train_predictions.csv"
MERGED_TEST_OUT = "data/answer/merged_predictions.csv"

METRICS_OUT = "data/answer/metrics.csv"


def parse_prediction(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value or value == "[]":
        return []
    return ast.literal_eval(value)


def run_regex_file(input_path: str, output_path: str):
    if input_path.endswith(".tsv"):
        df = read_train_dataset(input_path)
    else:
        df = read_test_dataset(input_path)

    rows = []
    for _, row in df.iterrows():
        pred = detect_pii(row["text"])
        item = {
            "text": row["text"],
            "prediction": str(pred),
        }
        if "id" in df.columns:
            item["id"] = row["id"]
        else:
            item["row_id"] = row["row_id"]
        rows.append(item)

    cols = ["id", "text", "prediction"] if "id" in df.columns else ["row_id", "text", "prediction"]
    out_df = pd.DataFrame(rows)[cols]
    out_df.to_csv(output_path, index=False)
    return out_df, df


def run_ner_file(input_path: str, output_path: str):
    if input_path.endswith(".tsv"):
        df = read_train_dataset(input_path)
    else:
        df = read_test_dataset(input_path)

    out_df = predict_dataframe(df)
    out_df.to_csv(output_path, index=False)
    return out_df, df


def run_merge(regex_path: str, ner_path: str, output_path: str):
    regex_df = pd.read_csv(regex_path)
    ner_df = pd.read_csv(ner_path)

    if len(regex_df) != len(ner_df):
        raise ValueError(
            f"Different row counts: regex={len(regex_df)}, ner={len(ner_df)}"
        )

    key = "id" if "id" in regex_df.columns else "row_id"
    if key in regex_df.columns and key in ner_df.columns:
        if regex_df[key].astype(str).tolist() != ner_df[key].astype(str).tolist():
            raise ValueError(f"{key} columns do not match")

    rows = []
    for (_, r_row), (_, n_row) in zip(regex_df.iterrows(), ner_df.iterrows()):
        regex_pred = parse_prediction(r_row["prediction"])
        ner_pred = parse_prediction(n_row["prediction"])
        merged = merge_predictions(regex_pred, ner_pred)

        item = {
            "text": r_row["text"],
            "prediction": str(merged),
        }
        if key in regex_df.columns:
            item[key] = r_row[key]
        rows.append(item)

    cols = [key, "text", "prediction"] if key in regex_df.columns else ["text", "prediction"]
    out_df = pd.DataFrame(rows)[cols]
    out_df.to_csv(output_path, index=False)
    return out_df


def append_metric(rows, model_name: str, dataset_name: str, pred_df: pd.DataFrame, gold_df: pd.DataFrame):
    if "target" not in gold_df.columns:
        return

    preds = pred_df["prediction"].apply(parse_prediction).tolist()
    targets = gold_df["target"].tolist()
    metrics = compute_metrics(preds, targets)

    rows.append(
        {
            "model": model_name,
            "dataset": dataset_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "micro_f1": metrics["micro_f1"],
        }
    )


def prepare_command(_args):
    ensure_dirs()
    train_df = read_train_dataset(TRAIN_PATH)
    test_df = read_test_dataset(TEST_PATH)
    save_processed(train_df, PROCESSED_TRAIN)
    save_processed(test_df, PROCESSED_TEST)
    print("Prepared data")


def regex_command(args):
    ensure_dirs()
    input_path = args.input
    output_path = args.output
    run_regex_file(input_path, output_path)
    print(f"Saved: {output_path}")


def ner_train_command(_args):
    ensure_dirs()
    train_df = read_train_dataset(TRAIN_PATH)
    train_ner(train_df, epochs=2, batch_size=8)
    print("NER trained on full train dataset")


def ner_predict_command(args):
    ensure_dirs()
    _, _ = run_ner_file(args.input, args.output)
    print(f"Saved: {args.output}")


def merge_command(_args):
    ensure_dirs()
    run_merge(REGEX_TEST_OUT, NER_TEST_OUT, MERGED_TEST_OUT)
    print(f"Saved: {MERGED_TEST_OUT}")


def all_command(_args):
    ensure_dirs()

    train_df = read_train_dataset(TRAIN_PATH)
    test_df = read_test_dataset(TEST_PATH)

    save_processed(train_df, PROCESSED_TRAIN)
    save_processed(test_df, PROCESSED_TEST)

    train_ner(train_df, epochs=2, batch_size=8)

    regex_train_df, gold_train_df = run_regex_file(TRAIN_PATH, REGEX_TRAIN_OUT)
    regex_test_df, _ = run_regex_file(TEST_PATH, REGEX_TEST_OUT)

    ner_train_df, _ = run_ner_file(TRAIN_PATH, NER_TRAIN_OUT)
    ner_test_df, _ = run_ner_file(TEST_PATH, NER_TEST_OUT)

    merged_train_df = run_merge(REGEX_TRAIN_OUT, NER_TRAIN_OUT, MERGED_TRAIN_OUT)
    merged_test_df = run_merge(REGEX_TEST_OUT, NER_TEST_OUT, MERGED_TEST_OUT)

    metric_rows = []
    append_metric(metric_rows, "regex", "train", regex_train_df, gold_train_df)
    append_metric(metric_rows, "ner", "train", ner_train_df, gold_train_df)
    append_metric(metric_rows, "merge", "train", merged_train_df, gold_train_df)

    save_metrics(metric_rows, METRICS_OUT)

    print(f"Saved: {REGEX_TRAIN_OUT}")
    print(f"Saved: {REGEX_TEST_OUT}")
    print(f"Saved: {NER_TRAIN_OUT}")
    print(f"Saved: {NER_TEST_OUT}")
    print(f"Saved: {MERGED_TRAIN_OUT}")
    print(f"Saved: {MERGED_TEST_OUT}")
    print(f"Saved: {METRICS_OUT}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("prepare")

    p_regex = subparsers.add_parser("regex")
    p_regex.add_argument("--input", required=True)
    p_regex.add_argument("--output", required=True)

    subparsers.add_parser("ner_train")

    p_ner_pred = subparsers.add_parser("ner_predict")
    p_ner_pred.add_argument("--input", required=True)
    p_ner_pred.add_argument("--output", required=True)

    subparsers.add_parser("merge")
    subparsers.add_parser("all")

    return parser


if __name__ == "__main__":
    parser = build_parser()
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
