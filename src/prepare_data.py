import ast
import pandas as pd


def safe_parse_target(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value or value == "[]":
        return []
    return ast.literal_eval(value)


def read_train_dataset(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].apply(safe_parse_target)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


def read_test_dataset(path: str):
    df = pd.read_csv(path, dtype=str)
    df["text"] = df["text"].fillna("").astype(str)
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = df.index
    return df


def save_processed(df, path: str):
    df.to_csv(path, index=False)


def spans_to_bio_by_offsets(offset_mapping, spans):
    tags = []
    for start, end in offset_mapping:
        if start == end:
            tags.append("O")
            continue

        tag = "O"
        for ent_start, ent_end, label in spans:
            if start >= ent_start and start < ent_end:
                if start == ent_start:
                    tag = f"B-{label}"
                else:
                    tag = f"I-{label}"
                break
        tags.append(tag)
    return tags
