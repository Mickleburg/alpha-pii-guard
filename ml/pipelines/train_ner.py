# File: ml/pipelines/train_ner.py
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score


def load_processed_data(split_path: Path) -> List[Dict[str, Any]]:
    """Load processed JSON data."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(labels_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load label list and create mappings."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    return label2id, id2label


def prepare_dataset(data: List[Dict[str, Any]], label2id: Dict[str, int]) -> Dataset:
    """Convert processed samples to HuggingFace Dataset format."""
    features = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    o_label_id = label2id["O"]

    for item in data:
        features["input_ids"].append(item["input_ids"])
        features["attention_mask"].append(item["attention_mask"])

        label_ids = []
        for label in item["labels"]:
            if label == -100:
                label_ids.append(-100)
            elif isinstance(label, int):
                label_ids.append(label)
            else:
                label_ids.append(label2id.get(label, o_label_id))
        features["labels"].append(label_ids)

    return Dataset.from_dict(features)


def build_compute_metrics(id2label: Dict[int, str]):
    """Build Trainer-compatible metric function with captured id2label."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        if hasattr(eval_pred, "predictions"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        predictions = np.argmax(predictions, axis=2)

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for pred_seq, label_seq in zip(predictions, labels):
            seq_labels = []
            seq_preds = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                seq_labels.append(id2label[int(label_id)])
                seq_preds.append(id2label[int(pred_id)])

            true_labels.append(seq_labels)
            true_predictions.append(seq_preds)

        return {
            "precision": float(precision_score(true_labels, true_predictions)),
            "recall": float(recall_score(true_labels, true_predictions)),
            "f1": float(f1_score(true_labels, true_predictions)),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="DeepPavlov/rubert-base-cased",
        help="HuggingFace model name",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml/models/ner",
        help="Output directory for model",
    )
    args = parser.parse_args()

    processed_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training NER model: {args.model_name}")
    print(f"Output directory: {output_dir}")

    print("\nLoading labels...")
    label2id, id2label = load_labels(processed_dir / "labels.json")
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")

    print("\nLoading labels...")
    label2id, id2label = load_labels(processed_dir / "labels.json")
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")

    tokenizer_meta_path = processed_dir / "tokenizer_name.json"
    if tokenizer_meta_path.exists():
        with open(tokenizer_meta_path, "r", encoding="utf-8") as f:
            tokenizer_meta = json.load(f)
        prepared_with = tokenizer_meta.get("model_name")
        if prepared_with and prepared_with != args.model_name:
            raise ValueError(
                f"Processed data was created with tokenizer '{prepared_with}', "
                f"but training uses '{args.model_name}'. "
                f"Re-run prepare_data.py with the same --model_name."
            )

    print("\nLoading training data...")
    train_data = load_processed_data(processed_dir / "train.json")
    val_data = load_processed_data(processed_dir / "val.json")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, label2id)
    val_dataset = prepare_dataset(val_data, label2id)

    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        dataloader_pin_memory=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nTraining completed!")
    train_loss = float(train_result.metrics.get("train_loss", 0.0))
    print(f"Train loss: {train_loss:.4f}")

    print("\nRunning final evaluation...")
    eval_result = trainer.evaluate()

    eval_loss = float(eval_result.get("eval_loss", 0.0))
    eval_f1 = float(eval_result.get("eval_f1", 0.0))
    eval_precision = float(eval_result.get("eval_precision", 0.0))
    eval_recall = float(eval_result.get("eval_recall", 0.0))

    print(f"Validation loss: {eval_loss:.4f}")
    print(f"Validation F1: {eval_f1:.4f}")
    print(f"Validation Precision: {eval_precision:.4f}")
    print(f"Validation Recall: {eval_recall:.4f}")

    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_f1": eval_f1,
                "eval_precision": eval_precision,
                "eval_recall": eval_recall,
                "num_labels": num_labels,
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nModel training complete! Saved to {output_dir}")


if __name__ == "__main__":
    main()
