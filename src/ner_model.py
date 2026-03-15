import os
import ast
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.labels import LABELS
from src.prepare_data import spans_to_bio_by_offsets


MODEL_NAME = "cointegrated/rubert-tiny2"
MODEL_DIR = "data/processed/ner_model"
MAX_LENGTH = 256
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BIO_LABELS = ["O"]
for label in LABELS:
    BIO_LABELS.append(f"B-{label}")
    BIO_LABELS.append(f"I-{label}")

TAG2ID = {tag: i for i, tag in enumerate(BIO_LABELS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}


class TokenDataset(Dataset):
    def __init__(self, texts: List[str], targets: List[List[Tuple[int, int, str]]], tokenizer):
        self.items = []

        for text, spans in zip(texts, targets):
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
            )

            bio_tags = spans_to_bio_by_offsets(enc["offset_mapping"], spans)
            labels = []

            for offset, tag in zip(enc["offset_mapping"], bio_tags):
                if offset[0] == offset[1]:
                    labels.append(-100)
                else:
                    labels.append(TAG2ID[tag])

            self.items.append(
                {
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABELS),
        id2label=ID2TAG,
        label2id=TAG2ID,
    )
    return tokenizer, model


def train_ner(train_df: pd.DataFrame, epochs: int = 2, batch_size: int = 8):
    set_seed(SEED)
    tokenizer, model = build_model()

    dataset = TokenDataset(
        texts=train_df["text"].tolist(),
        targets=train_df["target"].tolist(),
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        do_eval=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


def load_ner():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model


def clean_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    spans = sorted(spans, key=lambda x: (x[0], x[1], x[2]))
    result = []
    for span in spans:
        if span[0] >= span[1]:
            continue
        if not result or span[0] >= result[-1][1]:
            result.append(span)
    return result


def predict_one(text: str, tokenizer, model) -> List[Tuple[int, int, str]]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = enc.pop("offset_mapping")[0].tolist()

    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0].detach().cpu().numpy()

    pred_ids = logits.argmax(axis=-1).tolist()
    pred_tags = [ID2TAG[i] for i in pred_ids]

    spans = []
    current_start = None
    current_end = None
    current_label = None

    for tag, (start, end) in zip(pred_tags, offset_mapping):
        if start == end:
            continue

        if tag == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_start, current_end, current_label = None, None, None
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_start, current_end, current_label = start, end, label
        else:
            if current_label == label and current_end is not None and start <= current_end:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_start, current_end, current_label = start, end, label

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return clean_spans(spans)


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer, model = load_ner()
    rows = []

    for _, row in df.iterrows():
        text = row["text"]
        prediction = predict_one(text, tokenizer, model)

        out = {
            "text": text,
            "prediction": str(prediction),
        }

        if "id" in row.index:
            out["id"] = row["id"]
        else:
            out["row_id"] = row["row_id"]

        rows.append(out)

    cols = ["id", "text", "prediction"] if "id" in df.columns else ["row_id", "text", "prediction"]
    return pd.DataFrame(rows)[cols]
