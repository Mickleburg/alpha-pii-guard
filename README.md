# Alpha PII Guard

Проект решает задачу **Named Entity Recognition (NER)** для поиска чувствительных данных в русскоязычных текстах.

Модель должна:
- находить точные символьные границы сущностей;
- определять их категорию;
- возвращать результат в формате списка спанов: `(start, end, label)`.

Оценка качества:
- **Micro-Averaged F1-Score**;
- засчитывается только **строгое совпадение** границ сущности и её категории.

Запрещено использовать LLM через внешние API с промптами.  
Цель решения — получить качественное извлечение сущностей при минимальном времени обработки текста.

## Архитектура

Проект состоит из нескольких этапов:

1. **Подготовка данных**
   - чтение train/test данных;
   - преобразование разметки в BIO-формат.

2. **NER-модель**
   - обучение token-classification модели на базе BERT;
   - основная рекомендуемая модель: `DeepPavlov/rubert-base-cased`.

3. **Regex-детектор**
   - извлечение сущностей, которые удобно ловить регулярными выражениями.

4. **Слияние предсказаний**
   - объединение результатов NER и regex.

5. **Формирование submission**
   - преобразование финальных предсказаний в CSV нужного формата.

## Структура проекта

```text
project/
├── data/
│   ├── raw/
│   │   ├── train_dataset.tsv
│   │   └── private_test_dataset.csv
│   ├── processed/
│   └── answer/
├── src/
│   ├── __init__.py
│   ├── labels.py
│   ├── prepare_data.py
│   ├── regex_detector.py
│   ├── ner_model.py
│   ├── merge_predictions.py
│   ├── evaluate.py
│   └── utils.py
├── main.py
├── make_submissions.py
└── requirements.txt
```

## Установка

### Требования:

- Python 3.10+;

- `venv`;

- зависимости из `requirements.txt`.

### 1. Создать виртуальное окружение

```bash
python -m venv .venv
```

### 2. Активировать окружение

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Подготовка данных

Положите исходные файлы сюда:

```text
data/raw/train_dataset.tsv
data/raw/private_test_dataset.csv
```

После этого выполните:
```bash
python main.py prepare
```

## Обучение модели

Рекомендуемая модель:
```bash
python main.py ner_train --model-name deeppavlov
```

Если в вашей версии `src/ner_model.py` поддерживаются другие алиасы моделей, их можно передавать так же через аргумент `--model-name`, например:

```bash
python main.py ner_train --model-name base
python main.py ner_train --model-name tiny
python main.py ner_train --model-name large
```

Если нужно явно задать параметры:

```bash
python main.py ner_train \
  --model-name deeppavlov \
  --batch-size 1 \
  --max-len 256 \
  --epochs 3 \
  --learning-rate 2e-5
```

## Инференс и сборка результата

NER-предсказания
```bash
python main.py ner_predict
```

Regex-предсказания
```bash
python main.py regex
```

Объединение предсказаний
```bash
python main.py merge
```

Построение submission
```bash
python make_submissions.py data/answer/merged_predictions.csv
```

По умолчанию будет создан файл:
```text
data/answer/submissions.csv
```

Формат итогового файла:
```text
id,Prediction
0,[]
1,"[(107, 124, 'Номер телефона')]"
```

## Полный пайплайн

Пошагово:
```bash
python main.py prepare
python main.py ner_train --model-name deeppavlov
python main.py ner_predict
python main.py regex
python main.py merge
python make_submissions.py data/answer/merged_predictions.csv
```

Если ваш `main.py` поддерживает полный запуск одной командой:
```bash
python main.py all --model-name deeppavlov
python make_submissions.py data/answer/merged_predictions.csv
```

## Результаты
Финальный submission нужно загружать в формате:

- `id`

- `Prediction`

где `Prediction` — это строковое представление списка сущностей:

- `[]`, если сущностей нет;

- `[(start, end, label), ...]`, если сущности найдены.

## Результаты экспериментов kaggle score

- **Regex only**: `0.59950`

- **NER only**: `0.96727`

- **Merg**e: `0.95247`
