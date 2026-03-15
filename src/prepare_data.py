import ast
import pandas as pd


def safe_parse_target(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value or value == "[]" or value == "empty":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    result = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, label = item[0], item[1], item[2]
            result.append((int(start), int(end), str(label)))
    return result


def read_train_dataset(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]

    if "text" not in df.columns:
        raise ValueError(f"Column 'text' not found in train file. Columns: {df.columns.tolist()}")
    if "target" not in df.columns:
        raise ValueError(f"Column 'target' not found in train file. Columns: {df.columns.tolist()}")

    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].apply(safe_parse_target)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


def read_test_dataset(path: str):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]

    if "text" not in df.columns:
        raise ValueError(f"Column 'text' not found in test file. Columns: {df.columns.tolist()}")

    df["text"] = df["text"].fillna("").astype(str)

    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = df.index

    return df


def save_processed(df, path: str):
    df.to_csv(path, index=False, encoding="utf-8")


def spans_to_bio_by_offsets(offset_mapping, spans):
    tags = []

    for token_start, token_end in offset_mapping:
        if token_start == token_end:
            tags.append("O")
            continue

        tag = "O"
        for ent_start, ent_end, label in spans:
            if token_start >= ent_start and token_start < ent_end:
                if token_start == ent_start:
                    tag = f"B-{label}"
                else:
                    tag = f"I-{label}"
                break

        tags.append(tag)

    return tags
