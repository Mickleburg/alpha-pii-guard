# File: ml/pipelines/train_ner.py
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.metrics import classification_report

def load_processed_data(split_path: Path) -> List[Dict]:
    """Load processed JSON data."""
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_labels(labels_path: Path) -> Dict:
    """Load label list and create mappings."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    
    return label2id, id2label

def prepare_dataset(data: List[Dict], label2id: Dict) -> Dataset:
    """Convert data to HuggingFace Dataset format."""
    features = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for item in data:
        features['input_ids'].append(item['input_ids'])
        features['attention_mask'].append(item['attention_mask'])
        
        # Convert string labels to IDs
        label_ids = [label2id.get(label, label2id['O']) for label in item['labels']]
        features['labels'].append(label_ids)
    
    return Dataset.from_dict(features)

def compute_metrics(eval_pred, id2label):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding (-100)
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_seq.append(id2label[label])
                pred_seq_filtered.append(id2label[pred])
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)
    
    # Flatten for metrics
    true_flat = [label for seq in true_labels for label in seq]
    pred_flat = [label for seq in pred_labels for label in seq]
    
    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='micro', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--model_name', type=str, default='cointegrated/rubert-tiny2',
                      help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='ml/models/ner',
                      help='Output directory for model')
    args = parser.parse_args()
    
    # Paths
    processed_dir = Path('data/processed')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training NER model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    
    # Load labels
    print("\nLoading labels...")
    label2id, id2label = load_labels(processed_dir / 'labels.json')
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")
    
    # Load data
    print("\nLoading training data...")
    train_data = load_processed_data(processed_dir / 'train.json')
    val_data = load_processed_data(processed_dir / 'val.json')
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, label2id)
    val_dataset = prepare_dataset(val_data, label2id)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / 'checkpoints'),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none'
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id2label)
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    print("\nTraining completed!")
    print(f"Train loss: {train_result.metrics['train_loss']:.4f}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_result = trainer.evaluate()
    print(f"Validation loss: {eval_result['eval_loss']:.4f}")
    print(f"Validation F1: {eval_result['eval_f1']:.4f}")
    print(f"Validation Precision: {eval_result['eval_precision']:.4f}")
    print(f"Validation Recall: {eval_result['eval_recall']:.4f}")
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_loss': train_result.metrics['train_loss'],
            'eval_loss': eval_result['eval_loss'],
            'eval_f1': eval_result['eval_f1'],
            'eval_precision': eval_result['eval_precision'],
            'eval_recall': eval_result['eval_recall']
        }, f, indent=2)
    
    print(f"\nModel training complete! Saved to {output_dir}")

if __name__ == '__main__':
    main()
