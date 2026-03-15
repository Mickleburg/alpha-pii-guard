#!pip install -U transformers datasets pandas scikit-learn

import pandas as pd
import torch
import ast
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Фиксируем seed для воспроизводимости
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# Ячейка 2: основной код обучения с моделью DeepPavlov/rubert-base-cased

# ------------------------------
# 1. Загрузка данных (формат: text, target, entity)
# ------------------------------

df = pd.read_csv('data.tsv', sep='\t', header=None, names=['text', 'target', 'entity'])

# Функция безопасного парсинга target (список кортежей)
def parse_target(target_str):
    if pd.isna(target_str):
        return []
    target_str = str(target_str).strip()
    if target_str == '[]' or target_str == 'empty' or target_str == '':
        return []
    try:
        parsed = ast.literal_eval(target_str)
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, tuple) and len(item) == 3:
                    valid.append((int(item[0]), int(item[1]), str(item[2])))
                else:
                    print(f"Предупреждение: пропущен некорректный элемент {item} в строке: {target_str[:50]}...")
            return valid
        else:
            return []
    except Exception as e:
        print(f"Не удалось распарсить target: {target_str[:100]}... Ошибка: {e}")
        return []

df['entities'] = df['target'].apply(parse_target)
df = df.dropna(subset=['text']).reset_index(drop=True)

# Анализ распределения категорий (для информации)
all_categories = []
for entities in df['entities']:
    all_categories.extend([cat for _, _, cat in entities])
counter = Counter(all_categories)
print("Распределение категорий:")
for cat, count in counter.most_common():
    print(f"  {cat}: {count}")

categories = sorted(list(set(all_categories)))
print(f"\nВсего категорий: {len(categories)}")
if not categories:
    raise ValueError("В данных нет ни одной сущности с категориями. Обучение невозможно.")

# BIO-разметка
label_to_id = {'O': 0}
id_to_label = {0: 'O'}
for i, cat in enumerate(categories):
    label_to_id[f'B-{cat}'] = 2*i + 1
    label_to_id[f'I-{cat}'] = 2*i + 2
    id_to_label[2*i + 1] = f'B-{cat}'
    id_to_label[2*i + 2] = f'I-{cat}'

num_labels = len(label_to_id)
print(f"Всего меток (включая O): {num_labels}")

# ------------------------------
# 2. Токенизатор и выравнивание меток
# ------------------------------

# Используем более мощную модель
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(text, entities, tokenizer, label_to_id, max_length=512):
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors=None
    )
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    offsets = tokenized['offset_mapping']

    labels = [label_to_id['O']] * len(input_ids)

    for start_char, end_char, cat in entities:
        start_token_idx = None
        end_token_idx = None
        for i, (start_offset, end_offset) in enumerate(offsets):
            if start_offset is None:
                continue
            if start_offset <= start_char < end_offset:
                start_token_idx = i
            if start_offset < end_char <= end_offset:
                end_token_idx = i
            if start_token_idx is not None and end_token_idx is not None:
                break

        if start_token_idx is None or end_token_idx is None:
            continue

        for token_pos in range(start_token_idx, end_token_idx + 1):
            if token_pos == start_token_idx:
                labels[token_pos] = label_to_id[f'B-{cat}']
            else:
                labels[token_pos] = label_to_id[f'I-{cat}']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'offset_mapping': offsets,
        'true_entities': entities
    }

# ------------------------------
# 3. Создание датасета
# ------------------------------

data_records = []
for idx, row in df.iterrows():
    res = tokenize_and_align_labels(row['text'], row['entities'], tokenizer, label_to_id)
    data_records.append(res)

train_records, val_records = train_test_split(data_records, test_size=0.2, random_state=seed)

def convert_to_dataset(records):
    return Dataset.from_dict({
        'input_ids': [r['input_ids'] for r in records],
        'attention_mask': [r['attention_mask'] for r in records],
        'labels': [r['labels'] for r in records],
        'idx': list(range(len(records)))
    })

train_dataset = convert_to_dataset(train_records)
val_dataset = convert_to_dataset(val_records)

val_offsets = [r['offset_mapping'] for r in val_records]
val_true_entities = [r['true_entities'] for r in val_records]

# ------------------------------
# 4. Загрузка модели (большая)
# ------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id_to_label,
    label2id=label_to_id,
    ignore_mismatched_sizes=True
)

# ------------------------------
# 5. Метрика (строгое совпадение)
# ------------------------------

def spans_from_labels(labels, offset_mapping, id_to_label):
    entities = []
    i = 0
    while i < len(labels):
        label_id = labels[i]
        if label_id == 0:
            i += 1
            continue
        label_str = id_to_label[label_id]
        if label_str.startswith('B-'):
            cat = label_str[2:]
            start_offset = offset_mapping[i][0]
            j = i + 1
            while j < len(labels):
                next_label = id_to_label.get(labels[j], 'O')
                if next_label == f'I-{cat}':
                    j += 1
                else:
                    break
            end_offset = offset_mapping[j-1][1]
            entities.append((start_offset, end_offset, cat))
            i = j
        else:
            i += 1
    return entities

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_spans_list = []
    pred_spans_list = []

    for i in range(len(predictions)):
        pred = predictions[i]
        label = labels[i]

        active_indices = label != -100
        pred_filtered = pred[active_indices]

        offsets = val_offsets[i]
        offsets_filtered = [offsets[j] for j in range(len(offsets)) if label[j] != -100]

        pred_entities = spans_from_labels(pred_filtered, offsets_filtered, id_to_label)
        true_entities = val_true_entities[i]

        pred_spans_list.append(pred_entities)
        true_spans_list.append(true_entities)

    tp = fp = fn = 0
    for true_spans, pred_spans in zip(true_spans_list, pred_spans_list):
        true_set = set(true_spans)
        pred_set = set(pred_spans)
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}

# ------------------------------
# 6. Обучение (с early stopping)
# ------------------------------

training_args = TrainingArguments(
    output_dir='./ner_model',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,        # уменьшаем batch из-за размера модели
    per_device_eval_batch_size=8,
    num_train_epochs=10,                   # больше эпох
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,                              # включаем mixed precision для экономии памяти
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # остановка, если f1 не растёт 3 эпохи
)

trainer.train()

# Сохраняем финальную модель
trainer.save_model('./ner_model_final')
tokenizer.save_pretrained('./ner_model_final')

print("Обучение завершено, модель сохранена в './ner_model_final'")