import argparse
import ast
import os
from typing import List, Tuple, Any

import pandas as pd


def parse_prediction(value: Any) -> List[Tuple[int, int, str]]:
    """
    Безопасно парсим prediction из CSV.
    Ожидаем строку вида:
    "[(0, 10, 'ФИО'), (15, 25, 'Email')]"
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def normalize_spans(spans: List[Any]) -> List[Tuple[int, int, str]]:
    """
    Нормализуем spans к формату:
    [(start, end, label), ...]
    """
    result = []

    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue

        start, end, label = item

        try:
            start = int(start)
            end = int(end)
            label = str(label)
        except Exception:
            continue

        if start < 0 or end <= start:
            continue

        result.append((start, end, label))

    return sorted(set(result), key=lambda x: (x[0], x[1], x[2]))


def detect_prediction_column(df: pd.DataFrame) -> str:
    """
    Ищем колонку с предсказаниями.
    Поддерживаем и 'prediction', и 'Prediction'.
    """
    if "prediction" in df.columns:
        return "prediction"
    if "Prediction" in df.columns:
        return "Prediction"
    raise ValueError("Input CSV must contain 'prediction' or 'Prediction' column")


def build_submission(input_path: str, output_path: str | None = None) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str)

    pred_col = detect_prediction_column(df)

    predictions = []
    for value in df[pred_col]:
        spans = parse_prediction(value)
        spans = normalize_spans(spans)
        predictions.append(str(spans))

    if "id" in df.columns:
        ids = df["id"]
    else:
        ids = pd.Series(range(len(df)), name="id")

    submission = pd.DataFrame({
        "id": ids,
        "Prediction": predictions,
    })

    if output_path is None:
        input_dir = os.path.dirname(input_path) or "."
        output_path = os.path.join(input_dir, "submissions.csv")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Submission saved to: {output_path}")
    print(f"Rows: {len(submission)}")

    if len(submission) > 0:
        print("Sample row:")
        print(submission.iloc[0].to_dict())

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build submissions.csv in expected format: id,Prediction"
    )

    parser.add_argument(
        "input_path",
        help="Path to input predictions CSV"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output submissions.csv (default: next to input file)"
    )

    args = parser.parse_args()
    build_submission(args.input_path, args.output)
