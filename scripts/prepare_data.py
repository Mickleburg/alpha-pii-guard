# File: scripts/prepare_data.py
import os
import ast
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from ml.config.labels import strip_bio

random.seed(42)

def parse_target_column(target_str: str) -> List[Tuple[int, int, str]]:
    """Parse target column from string representation to list of tuples."""
    if pd.isna(target_str) or target_str == '[]' or target_str == '' or target_str == 'empty':
        return []
    try:
        parsed = ast.literal_eval(target_str)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []

def normalize_spans(spans):
    result = []
    for start, end, label in spans:
        result.append((int(start), int(end), strip_bio(label)))
    return result


def parse_entity_column(entity_str: str) -> List[str]:
    """Parse entity column for validation purposes."""
    if pd.isna(entity_str) or entity_str == 'empty' or entity_str == '[]' or entity_str == '':
        return []
    try:
        parsed = ast.literal_eval(entity_str)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []

def collect_all_categories(df: pd.DataFrame) -> List[str]:
    """Collect all unique categories from the dataset."""
    categories = set()
    for idx, row in df.iterrows():
        spans = parse_target_column(row['target'])
        for start, end, category in spans:
            categories.add(category)
    return sorted(list(categories))

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
    parser = argparse.ArgumentParser(description='Prepare dataset for NER training')
    parser.add_argument('--input', type=str, default='data/raw/train_dataset.tsv',
                      help='Input TSV file path')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Output directory for processed data')
    parser.add_argument('--model_name', type=str, default='DeepPavlov/rubert-base-cased',
                      help='Tokenizer model name')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                      help='Validation split ratio')
    args = parser.parse_args()
    
    # Paths
    raw_data_path = Path(args.input)
    processed_dir = Path(args.output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    model_name = args.model_name
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read dataset
    print(f"Reading dataset from {raw_data_path}")
    df = pd.read_csv(raw_data_path, sep='\t')
    print(f"Loaded {len(df)} rows")
    
    # Validate columns
    required_columns = {'text', 'target', 'entity'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        exit(1)
    
    # Parse target column
    print("Parsing target annotations...")
    df['target_parsed'] = df['target'].apply(parse_target_column).apply(normalize_spans)
    
    # Filter out rows with valid text
    df = df[df['text'].notna()].copy()
    df['text'] = df['text'].astype(str)
    
    # Collect all categories
    print("Collecting categories...")
    all_categories = collect_all_categories(df)
    print(f"Found {len(all_categories)} unique categories")
    
    # Statistics
    entity_counts = defaultdict(int)
    rows_with_entities = 0
    rows_without_entities = 0
    invalid_rows = 0
    
    for idx, row in df.iterrows():
        spans = row['target_parsed']
        if spans is None or (isinstance(spans, list) and len(spans) == 0):
            rows_without_entities += 1
        else:
            rows_with_entities += 1
            for _, _, cat in spans:
                entity_counts[cat] += 1
    
    print("\nDataset statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with entities: {rows_with_entities}")
    print(f"  Rows without entities: {rows_without_entities}")
    print(f"\nTop 20 categories by frequency:")
    for cat, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cat}: {count}")
    
    # Create label list
    all_labels = ['O'] + [f'B-{cat}' for cat in all_categories] + [f'I-{cat}' for cat in all_categories]
    label_path = processed_dir / 'labels.json'
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_labels)} labels to {label_path}")

    tokenizer_meta_path = processed_dir / 'tokenizer_name.json'
    with open(tokenizer_meta_path, 'w', encoding='utf-8') as f:
        json.dump({"model_name": args.model_name}, f, ensure_ascii=False, indent=2)
    print(f"Saved tokenizer metadata to {tokenizer_meta_path}")
    
    # Shuffle and split
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)
    
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
            char_spans = row['target_parsed']
            
            if char_spans is None:
                char_spans = []
            
            try:
                tokenized = tokenize_and_align(text, char_spans, tokenizer)
                tokenized_data.append(tokenized)
            except Exception as e:
                print(f"Warning: Failed to process row {idx}: {e}")
                continue
        
        # Save tokenized
        tokenized_path = processed_dir / f'{split_name}.json'
        with open(tokenized_path, 'w', encoding='utf-8') as f:
            json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(tokenized_data)} tokenized samples to {tokenized_path}")
        
        # Raw version (for evaluation)
        raw_data = []
        for idx in split_indices:
            row = df.iloc[idx]
            spans = row['target_parsed']
            if spans is None:
                spans = []
            
            # Convert to list of lists for JSON serialization
            spans_list = [[start, end, cat] for start, end, cat in spans]
            
            raw_data.append({
                'text': row['text'],
                'spans': spans_list
            })
        
        raw_path = processed_dir / f'{split_name}_raw.jsonl'
        with open(raw_path, 'w', encoding='utf-8') as f:
            for item in raw_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(raw_data)} raw samples to {raw_path}")
    
    print("\nData preparation complete!")

if __name__ == '__main__':
    main()
