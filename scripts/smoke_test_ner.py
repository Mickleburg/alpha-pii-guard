# File: scripts/smoke_test_ner.py
"""
Smoke test for NER model inference.

Runs predictions on hardcoded Russian examples to verify:
- Model loads correctly
- Predictions run without errors
- Output format is correct
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipelines.infer_ner import NERModel

# Test cases with expected PII types
TEST_CASES = [
    {
        "text": "Мой паспорт серии 4510 номер 654321, позвоните мне.",
        "description": "Russian passport (series + number)"
    },
    {
        "text": "Свяжитесь со мной по телефону +7 900 123-45-67 или 8 (495) 123-4567.",
        "description": "Phone numbers (multiple formats)"
    },
    {
        "text": "Отправьте документы на ivan.petrov@example.com",
        "description": "Email address"
    },
    {
        "text": "Меня зовут Иванов Иван Петрович, дата рождения 15.03.1985",
        "description": "Full name + date of birth"
    },
    {
        "text": "Мой СНИЛС 123-456-789 12, ИНН 771234567890",
        "description": "SNILS + INN"
    },
    {
        "text": "Карта номер 4276 1234 5678 9012, счет 40817810099910004312",
        "description": "Bank card + account number"
    },
    {
        "text": "Я живу по адресу Москва, улица Ленина, дом 10, квартира 25",
        "description": "Address"
    },
    {
        "text": "Это обычный текст без персональных данных о погоде и природе.",
        "description": "No PII (negative case)"
    }
]

def format_prediction(text: str, entities: List[Tuple[int, int, str]]) -> str:
    """Format prediction for display."""
    if not entities:
        return "  [No entities detected]"
    
    lines = []
    for start, end, category in entities:
        extracted = text[start:end]
        lines.append(f"  [{start:3d}:{end:3d}] {category:<30} → \"{extracted}\"")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Smoke test for NER model')
    parser.add_argument('--model_dir', type=str, default='ml/models/ner',
                       help='Path to model directory')
    parser.add_argument('--fail_on_empty', action='store_true',
                       help='Exit with error if all predictions are empty')
    args = parser.parse_args()
    
    print("="*80)
    print("NER MODEL SMOKE TEST")
    print("="*80)
    print(f"Model directory: {args.model_dir}")
    print(f"Fail on empty: {args.fail_on_empty}")
    print()
    
    # Check if model exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"❌ ERROR: Model directory not found: {model_path}")
        print("Run training first: python ml/pipelines/train_ner.py")
        sys.exit(1)
    
    # Load model
    try:
        print("Loading NER model...")
        model = NERModel(model_dir=args.model_dir)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Run predictions
    print("-"*80)
    print("RUNNING PREDICTIONS")
    print("-"*80)
    
    all_empty = True
    total_entities = 0
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        text = test_case['text']
        description = test_case['description']
        
        print(f"\n[Test {idx}] {description}")
        print(f"Text: {text}")
        print("Predictions:")
        
        try:
            entities = model.predict(text)
            
            if entities:
                all_empty = False
                total_entities += len(entities)
            
            formatted = format_prediction(text, entities)
            print(formatted)
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Total entities detected: {total_entities}")
    
    if all_empty:
        print("⚠️  WARNING: All predictions are empty")
        if args.fail_on_empty:
            print("❌ FAIL: Exiting with error (--fail_on_empty is set)")
            sys.exit(1)
    else:
        print(f"✓ Success: {total_entities} entities detected across test cases")
    
    print("\n✓ Smoke test completed successfully")
    sys.exit(0)

if __name__ == '__main__':
    main()
