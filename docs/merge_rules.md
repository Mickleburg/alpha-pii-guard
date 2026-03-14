# Merge Rules Documentation

## Overview

This document describes the merge logic for combining regex and NER predictions in alpha-pii-guard.

## Why Unified Format?

The system uses multiple detection approaches:
- **Regex**: Rule-based patterns for structured entities
- **NER**: Context-aware ML model for unstructured entities

To enable accurate evaluation and intelligent merging, both detectors must return the same format:
```python
List[Tuple[int, int, str]]  # (start, end, label)
Where:

start: character offset where entity begins (inclusive)

end: character offset where entity ends (exclusive)

label: normalized category name without BIO prefixes

Label Normalization
BIO Tagging
NER models typically use BIO (Beginning-Inside-Outside) tagging:

B-Email: Beginning of Email entity

I-Email: Inside (continuation) of Email entity

O: Outside any entity

For external API and metrics, these must be normalized to final category names:

text
B-Email, I-Email → Email
B-Номер телефона, I-Номер телефона → Номер телефона
B-ФИО, I-ФИО → ФИО
Implementation
Normalization is handled by ml/merge/label_map.py:

normalize_label(label): Strip BIO prefix and return canonical name

BIO_TO_FINAL_LABEL: Explicit mapping dictionary

FINAL_LABELS: Set of all 32 category names

All entity outputs pass through normalization before returning to user.

Category Priorities
Not all entity types are equally suited to regex vs NER detection.

Regex-First Categories (Structured)
These entities follow consistent patterns and are best detected by rules:

Email

Номер телефона

Паспортные данные

Сведения об ИНН

СНИЛС клиента

Номер карты

Номер банковского счета

CVV/CVC

ПИН код

Дата окончания срока действия карты

Одноразовые коды

API ключи

Серия и номер вида на жительство

Водительское удостоверение

Временное удостоверение личности

Свидетельство о рождении

Содержимое магнитной полосы

Merge behavior: When regex and NER both detect overlapping spans in these categories, prefer regex result.

NER-First Categories (Context-Dependent)
These entities require context and are better detected by ML:

ФИО

Полный адрес

Место рождения

Гражданство и названия стран

Данные об организации/юридическом лице

Наименование банка

Имя держателя карты

Данные об автомобиле клиента

Разрешение на работу / визу

Кодовые слова

Merge behavior: When NER and regex both detect overlapping spans in these categories, prefer NER result.

Merge Algorithm
The merge process follows these steps:

1. Normalization
Validate entity boundaries (start < end, within text bounds)

Trim leading/trailing whitespace

Normalize labels (strip BIO prefixes)

Filter invalid entities

2. Deduplication
Group entities by (start, end, label)

For exact duplicates, keep entity with highest confidence score

Remove redundant entities

3. Conflict Resolution
When entities overlap, apply priority rules:

Exact same span + same label:

Keep one (arbitrary but deterministic)

Same label, nested spans:

Prefer more precise (shorter) span

Example: (0, 20, "ФИО") vs (5, 15, "ФИО") → keep (5, 15, "ФИО")

Different labels, overlapping:

Apply category priority (regex-first vs ner-first)

Use confidence scores as tie-breaker

For structured labels: prefer shorter spans

For context labels: prefer longer spans

No overlap:

Keep both entities

4. Sorting
Sort by (start, end, label) for deterministic output

Ensures consistent evaluation and reproducibility

Impact on Metrics
Strict Evaluation
The system uses strict span + category matching:

True Positive (TP): Prediction has exact (start, end, label) match in gold

False Positive (FP): Prediction has no exact match in gold

False Negative (FN): Gold entity has no exact match in predictions

Example:

python
Prediction: (0, 10, "Email")
Gold:       (0, 10, "Email")
Result:     TP = 1

Prediction: (0, 10, "Email")
Gold:       (0, 5, "Email")   # Wrong span
Result:     FP = 1, FN = 1

Prediction: (0, 10, "Email")
Gold:       (0, 10, "ФИО")    # Wrong label
Result:     FP = 1, FN = 1
Why Merging Matters
Without intelligent merging:

Duplicate detections (same entity from regex + NER) would inflate TP/FP

Conflicting labels would reduce precision

Sub-optimal spans would miss gold matches

With merging:

Duplicates deduplicated

Best detector chosen per category

Spans refined for precision

Usage
Running Evaluation
bash
# Evaluate all detectors
python -m ml.eval.evaluate_merge --input data/processed/test.tsv --mode all

# Evaluate specific detector
python -m ml.eval.evaluate_merge --mode regex
python -m ml.eval.evaluate_merge --mode ner
python -m ml.eval.evaluate_merge --mode merged

# Specify output path
python -m ml.eval.evaluate_merge --output results/eval.json
Running Tests
bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_merge_resolver.py
pytest tests/test_metrics_rules_ner.py
pytest tests/test_label_normalization.py

# Run with coverage
pytest --cov=ml tests/
Using Detectors in Code
python
from ml.pipelines.detect_merged import detect_regex, detect_ner, detect_merged

text = "Мой email: test@example.com, телефон +7 (999) 123-45-67"

# Regex only
regex_spans = detect_regex(text)
# [(11, 29, 'Email'), (40, 59, 'Номер телефона')]

# NER only
ner_spans = detect_ner(text)
# Output depends on NER model

# Merged (best of both)
merged_spans = detect_merged(text)
# Intelligent combination with conflict resolution
Configuration
Priority configuration in ml/merge/config.py:

python
# Change category priority
from ml.merge.config import CATEGORY_SOURCE_PRIORITY

# Make "Дата рождения" regex-first
CATEGORY_SOURCE_PRIORITY["Дата рождения"] = "regex"
Best Practices
Always normalize labels before comparing or evaluating

Use strict matching for evaluation (exact span + label)

Validate entity boundaries (trim whitespace, check bounds)

Prefer category priorities over raw scores when merging

Test determinism - same input should always produce same output

Document assumptions - when adding new categories, classify as structured or context-dependent

Troubleshooting
Problem: Metrics lower than expected

Check:

Are labels normalized consistently?

Are spans trimmed properly?

Are entity boundaries valid?

Is evaluation using strict matching?

Problem: Merge results favor wrong detector

Check:

Is category assigned correct priority in config?

Are confidence scores properly set?

Is conflict resolution logic correct for category type?

Problem: Non-deterministic results

Check:

Are entities sorted consistently?

Is tie-breaking logic deterministic?

Are dict iterations ordered?