import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# Загрузка обученной модели
# ------------------------------
model_path = './ner_model_final'  # путь к сохранённой модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

id2label = model.config.id2label  # словарь для преобразования ID меток в строковые категории

# ------------------------------
# Функция преобразования предсказаний в спаны
# ------------------------------
def spans_from_labels(labels, offset_mapping, id2label):
    """
    labels: список ID предсказанных меток (только для значимых токенов)
    offset_mapping: список кортежей (start, end) для тех же токенов
    """
    entities = []
    i = 0
    while i < len(labels):
        label_id = labels[i]
        label_str = id2label.get(label_id, 'O')
        if label_str == 'O':
            i += 1
            continue
        if label_str.startswith('B-'):
            cat = label_str[2:]
            start_offset = offset_mapping[i][0]
            j = i + 1
            while j < len(labels):
                next_label = id2label.get(labels[j], 'O')
                if next_label == f'I-{cat}':
                    j += 1
                else:
                    break
            end_offset = offset_mapping[j-1][1]
            entities.append((int(start_offset), int(end_offset), cat))
            i = j
        else:
            i += 1
    return entities

# ------------------------------
# Функция предсказания для одного текста
# ------------------------------
def predict_entities(text, tokenizer, model, id2label, max_length=512):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # Фильтруем специальные токены и паддинг
    mask = attention_mask[0].cpu().numpy().astype(bool)
    valid_indices = [i for i, (start, end) in enumerate(offset_mapping) if mask[i] and not (start == 0 and end == 0)]
    if not valid_indices:
        return []

    pred_labels = predictions[valid_indices]
    valid_offsets = [offset_mapping[i] for i in valid_indices]
    return spans_from_labels(pred_labels, valid_offsets, id2label)

# ------------------------------
# Загрузка входного CSV (должен содержать колонки 'id_text' и 'text')
# ------------------------------
df_input = pd.read_csv('input.csv')  # укажите имя вашего файла

# Убедимся, что колонка с текстом называется 'text', если нет – скорректируйте
# Если в файле колонка называется 'id_text' и 'text', то всё хорошо
if 'text' not in df_input.columns:
    # возможно колонка называется по-другому, например 'Текст'
    # тогда нужно её переименовать
    pass  # подстройте под ваш файл

# Применяем модель к каждому тексту
df_input['entities'] = df_input['text'].apply(lambda x: predict_entities(x, tokenizer, model, id2label))

# ------------------------------
# Формируем выходной DataFrame с колонками 'id' и 'Prediction'
# ------------------------------
# Если есть колонка 'id_text', используем её как id
if 'id_text' in df_input.columns:
    df_output = pd.DataFrame({
        'id': df_input['id_text'],
        'Prediction': df_input['entities']
    })
else:
    # иначе используем индекс строки (начиная с 0)
    df_output = pd.DataFrame({
        'id': range(len(df_input)),
        'Prediction': df_input['entities']
    })

# Сохраняем результат
df_output.to_csv('output_predictions.csv', index=False)
print("Результат сохранён в output_predictions.csv")