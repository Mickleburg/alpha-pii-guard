# File: scripts/prepare_data.py
import os
import ast
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

random.seed(42)

# Category mapping
CATEGORIES = [
    "ФИО",
    "Паспортные данные РФ",
    "СНИЛС",
    "ИНН",
    "Дата рождения",
    "Адрес",
    "Номер телефона",
    "Email",
    "Банковские данные"
]

def parse_entity_column(entity_str: str) -> List[Tuple[int, int, str]]:
    """Parse entity column from string representation to list of tuples."""
    if pd.isna(entity_str) or entity_str == '[]' or entity_str == '':
        return []
    try:
        parsed = ast.literal_eval(entity_str)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []

def char_spans_to_bio_labels(
    text: str,
    char_spans: List[Tuple[int, int, str]],
    tokens: List[str],
    offset_mapping: List[Tuple[int, int]]
) -> List[str]:
    """Convert character-level spans to BIO token-level labels."""
    labels = ['O'] * len(tokens)
    
    for start_char, end_char, category in char_spans:
        entity_started = False
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end:  # Special tokens
                continue
            
            # Token overlaps with entity
            if token_start < end_char and token_end > start_char:
                if not entity_started:
                    labels[idx] = f'B-{category}'
                    entity_started = True
                else:
                    labels[idx] = f'I-{category}'
    
    return labels

def tokenize_and_align(
    text: str,
    char_spans: List[Tuple[int, int, str]],
    tokenizer,
    max_length: int = 512
) -> Dict:
    """Tokenize text and align labels."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']
    
    labels = char_spans_to_bio_labels(text, char_spans, tokens, offset_mapping)
    
    return {
        'text': text,
        'tokens': tokens,
        'labels': labels,
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'char_spans': char_spans
    }

def main():
    # Paths
    raw_data_path = Path('data/raw/train_dataset.tsv')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    model_name = 'cointegrated/rubert-tiny2'
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read dataset
    print(f"Reading dataset from {raw_data_path}")
    df = pd.read_csv(raw_data_path, sep='\t')
    print(f"Loaded {len(df)} rows")
    
    # Parse entity column
    print("Parsing entity annotations...")
    df['entity_parsed'] = df['entity'].apply(parse_entity_column)
    
    # Filter out rows with valid text
    df = df[df['text'].notna()].copy()
    df['text'] = df['text'].astype(str)
    
    # Statistics
    entity_counts = defaultdict(int)
    for entities in df['entity_parsed']:
        for _, _, cat in entities:
            entity_counts[cat] += 1
    
    print("\nEntity distribution:")
    for cat, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Shuffle and split
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    n_train = int(len(indices) * 0.7)
    n_val = int(len(indices) * 0.15)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    print(f"\nSplits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    # Process and save each split
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Tokenized version
        tokenized_data = []
        for idx in tqdm(split_indices, desc=f"Tokenizing {split_name}"):
            row = df.iloc[idx]
            text = row['text']
            char_spans = row['entity_parsed']
            
            tokenized = tokenize_and_align(text, char_spans, tokenizer)
            tokenized_data.append(tokenized)
        
        # Save tokenized
        tokenized_path = processed_dir / f'{split_name}.json'
        with open(tokenized_path, 'w', encoding='utf-8') as f:
            json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
        print(f"Saved tokenized data to {tokenized_path}")
        
        # Raw version (for rule-based and evaluation)
        raw_data = []
        for idx in split_indices:
            row = df.iloc[idx]
            raw_data.append({
                'text': row['text'],
                'entities': row['entity_parsed'],
                'entity_texts': ast.literal_eval(row['entity_texts']) if pd.notna(row['entity_texts']) else []
            })
        
        raw_path = processed_dir / f'{split_name}_raw.jsonl'
        with open(raw_path, 'w', encoding='utf-8') as f:
            for item in raw_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved raw data to {raw_path}")
    
    # Save label list
    all_labels = ['O'] + [f'B-{cat}' for cat in CATEGORIES] + [f'I-{cat}' for cat in CATEGORIES]
    label_path = processed_dir / 'labels.json'
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)
    print(f"\nSaved label list to {label_path}")
    
    print("\nData preparation complete!")

if __name__ == '__main__':
    main()
