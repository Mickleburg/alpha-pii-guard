# File: scripts/check_dataset.py
"""
Validate train_dataset.tsv for correctness.

Checks:
- Required columns exist
- Entity parsing with ast.literal_eval
- Tuple format validation
- Span boundaries
- Substring extraction
"""
import sys
import ast
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd

def parse_entity_safely(entity_str: str) -> List[Tuple[int, int, str]]:
    """Parse entity column safely."""
    if pd.isna(entity_str) or entity_str == '[]' or entity_str == '':
        return []
    try:
        parsed = ast.literal_eval(entity_str)
        if not isinstance(parsed, list):
            return None
        return parsed
    except (ValueError, SyntaxError, TypeError):
        return None

def validate_entity_tuple(entity_tuple) -> Tuple[bool, str]:
    """Validate single entity tuple format."""
    if not isinstance(entity_tuple, tuple):
        return False, "Not a tuple"
    
    if len(entity_tuple) != 3:
        return False, f"Expected 3 elements, got {len(entity_tuple)}"
    
    start, end, category = entity_tuple
    
    if not isinstance(start, int):
        return False, f"start is not int: {type(start)}"
    
    if not isinstance(end, int):
        return False, f"end is not int: {type(end)}"
    
    if not isinstance(category, str):
        return False, f"category is not str: {type(category)}"
    
    if start < 0:
        return False, f"start < 0: {start}"
    
    if end <= start:
        return False, f"end <= start: {start}, {end}"
    
    return True, "OK"

def validate_span_bounds(text: str, start: int, end: int) -> Tuple[bool, str]:
    """Check if span is within text bounds."""
    text_len = len(text)
    
    if start >= text_len:
        return False, f"start {start} >= text length {text_len}"
    
    if end > text_len:
        return False, f"end {end} > text length {text_len}"
    
    return True, "OK"

def main():
    dataset_path = Path('data/raw/train_dataset.tsv')
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print(f"Validating dataset: {dataset_path}")
    print("="*70)
    
    # Read dataset
    try:
        df = pd.read_csv(dataset_path, sep='\t')
    except Exception as e:
        print(f"ERROR: Failed to read TSV: {e}")
        sys.exit(1)
    
    print(f"Total rows: {len(df)}")
    
    # Check required columns
    required_columns = {'text', 'entity', 'entity_texts'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"✓ Required columns present: {required_columns}")
    
    # Validation statistics
    stats = {
        'valid_rows': 0,
        'invalid_rows': 0,
        'empty_entities': 0,
        'parse_errors': 0,
        'format_errors': 0,
        'bound_errors': 0,
        'total_entities': 0,
        'category_counts': defaultdict(int)
    }
    
    errors = []
    
    # Validate each row
    for idx, row in df.iterrows():
        text = row['text']
        entity_str = row['entity']
        
        # Check text
        if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
            stats['invalid_rows'] += 1
            errors.append(f"Row {idx}: Empty or invalid text")
            continue
        
        # Parse entities
        entities = parse_entity_safely(entity_str)
        
        if entities is None:
            stats['parse_errors'] += 1
            stats['invalid_rows'] += 1
            errors.append(f"Row {idx}: Failed to parse entity column")
            continue
        
        if len(entities) == 0:
            stats['empty_entities'] += 1
            stats['valid_rows'] += 1
            continue
        
        # Validate each entity
        row_valid = True
        for ent_idx, entity in enumerate(entities):
            # Validate tuple format
            is_valid, error_msg = validate_entity_tuple(entity)
            if not is_valid:
                stats['format_errors'] += 1
                errors.append(f"Row {idx}, Entity {ent_idx}: {error_msg}")
                row_valid = False
                continue
            
            start, end, category = entity
            
            # Validate bounds
            is_valid, error_msg = validate_span_bounds(text, start, end)
            if not is_valid:
                stats['bound_errors'] += 1
                errors.append(f"Row {idx}, Entity {ent_idx}: {error_msg}")
                row_valid = False
                continue
            
            # Extract and validate substring
            try:
                extracted = text[start:end]
                if len(extracted) == 0:
                    stats['bound_errors'] += 1
                    errors.append(f"Row {idx}, Entity {ent_idx}: Extracted empty substring")
                    row_valid = False
                    continue
            except Exception as e:
                stats['bound_errors'] += 1
                errors.append(f"Row {idx}, Entity {ent_idx}: Failed to extract substring: {e}")
                row_valid = False
                continue
            
            # Track category
            stats['category_counts'][category] += 1
            stats['total_entities'] += 1
        
        if row_valid:
            stats['valid_rows'] += 1
        else:
            stats['invalid_rows'] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Valid rows:        {stats['valid_rows']:>6} ({stats['valid_rows']/len(df)*100:.1f}%)")
    print(f"Invalid rows:      {stats['invalid_rows']:>6} ({stats['invalid_rows']/len(df)*100:.1f}%)")
    print(f"Empty entities:    {stats['empty_entities']:>6}")
    print(f"Parse errors:      {stats['parse_errors']:>6}")
    print(f"Format errors:     {stats['format_errors']:>6}")
    print(f"Bound errors:      {stats['bound_errors']:>6}")
    print(f"Total entities:    {stats['total_entities']:>6}")
    
    print("\n" + "-"*70)
    print("CATEGORY DISTRIBUTION")
    print("-"*70)
    for category, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1]):
        print(f"  {category:<35} {count:>6}")
    
    # Print errors (first 20)
    if errors:
        print("\n" + "-"*70)
        print("VALIDATION ERRORS (first 20)")
        print("-"*70)
        for error in errors[:20]:
            print(f"  {error}")
        
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    print("\n" + "="*70)
    
    # Exit with appropriate code
    if stats['invalid_rows'] > 0:
        print("❌ VALIDATION FAILED - Critical errors found")
        sys.exit(1)
    else:
        print("✓ VALIDATION PASSED - Dataset is valid")
        sys.exit(0)

if __name__ == '__main__':
    main()
