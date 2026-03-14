# File: docs/model_card.md
# Russian PII Detection NER Model

## Model Description

**Model name:** Russian PII NER  
**Base checkpoint:** cointegrated/rubert-tiny2  
**Task:** Named Entity Recognition (NER) for Personal Identifiable Information (PII) detection in Russian text  
**Language:** Russian (ru)

## Task Description

This model identifies and classifies Personal Identifiable Information (PII) in Russian text for real-time masking in LLM requests. The model detects 9 categories of sensitive information:

1. **ФИО** - Full name
2. **Паспортные данные РФ** - Russian passport series and number
3. **СНИЛС** - Russian social security number
4. **ИНН** - Taxpayer identification number
5. **Дата рождения** - Date of birth
6. **Адрес** - Address
7. **Номер телефона** - Phone number
8. **Email** - Email address
9. **Банковские данные** - Bank card/account numbers

## Dataset Statistics

- **Total samples:** 8,000
- **Train split:** 5,600 (70%)
- **Validation split:** 1,200 (15%)
- **Test split:** 1,200 (15%)
- **Format:** Character-level span annotations with category labels

## Training Hyperparameters

- **Base model:** cointegrated/rubert-tiny2
- **Learning rate:** 2e-5
- **Batch size:** 32
- **Epochs:** 5
- **Weight decay:** 0.01
- **Optimizer:** AdamW
- **Max sequence length:** 512 tokens
- **Label scheme:** BIO (Begin-Inside-Outside)

## Evaluation Metrics

**Evaluation methodology:** Strict span + category matching
- True Positive (TP): Predicted span matches ground truth exactly (start, end, and category)
- False Positive (FP): Predicted span not in ground truth
- False Negative (FN): Ground truth span not predicted

**Primary metric:** Micro-averaged F1-score across all categories

### Results

Results will be populated after running `ml/eval/eval_ner.py`:
- Micro-averaged Precision: TBD
- Micro-averaged Recall: TBD
- Micro-averaged F1: TBD
- Per-category breakdown: See `docs/eval_results.json`

## Inference Example

```python
from ml.pipelines.infer_ner import NERModel

# Load model
model = NERModel(model_dir="ml/models/ner/")

# Predict entities
text = "Мой паспорт серии 4510 номер 654321, телефон +7 900 123-45-67"
entities = model.predict(text)

# Output: [(16, 20, 'Паспортные данные РФ'), (27, 33, 'Паспортные данные РФ'), 
#          (43, 60, 'Номер телефона')]

# Batch prediction
texts = ["Текст 1", "Текст 2"]
batch_results = model.predict_batch(texts)
```

# Usage in Production

```python
from app.services.detect_entities import detect_entities, mask_text, demask_text

# Detect PII
text = "Позвоните мне на +7 900 123-45-67"
entities = detect_entities(text)

# Mask PII
masked_text, mapping = mask_text(text, entities)
# Output: "Позвоните мне на [PII_0]"

# Send masked_text to LLM...

# Demask response
response = "Хорошо, я позвоню на [PII_0]"
original_response = demask_text(response, mapping)
# Output: "Хорошо, я позвоню на +7 900 123-45-67"
```

## Performance
- Inference latency: <50ms per short text (target: <100ms)

- Model size: ~29MB (rubert-tiny2 base)

- Device: Supports both CPU and GPU inference

## Limitations
- Maximum sequence length: 512 tokens (longer texts handled via sliding window)

- Trained on specific Russian PII categories - may not generalize to other domains

- Rule-based patterns supplement NER for high-precision detection

- Performance depends on training data quality and coverage

## Model Architecture
- Type: Token classification (BERT-based)

- Base: cointegrated/rubert-tiny2 (12 layers, 312 hidden size, 12 attention heads)

- Classification head: Linear layer with num_labels=19 (9 categories × 2 BIO tags + O)

## Training Infrastructure
- Framework: HuggingFace Transformers 4.36+

- Backend: PyTorch 2.1+

- Evaluation: Strict span matching with micro-averaged metrics

## Citation
```text
@misc{russian-pii-ner-2026,
  title={Russian PII Detection NER Model},
  author={Alpha PII Guard Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Mickleburg/alpha-pii-guard}}
}
```

## License
This model is released for use in the Alfabank LLM hackathon competition.

## Contact
For questions or issues, please open an issue in the GitHub repository.