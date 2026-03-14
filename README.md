# PII NER Detection System

Production-grade система для детекции PII в русском тексте.

Проект объединяет три подхода:
- regex-детектор для точных правил;
- BERT-based NER для сложных сущностей;
- merge-слой для объединения результатов с приоритетом regex.

Финальный формат предсказаний:
`List[Tuple[int, int, str]]`

Где:
- `start_char` — включительная позиция начала;
- `end_char` — исключительная позиция конца;
- `category` — тип сущности.

---

## Назначение

Система ищет персональные и чувствительные данные в русском тексте и возвращает точные символьные координаты найденных сущностей.

Проект рассчитан на локальный запуск без внешних LLM API, что упрощает безопасное использование в production-сценариях.

---

## Архитектура

Пайплайн состоит из трех независимых частей.

### 1. Regex Detector

Regex-детектор отвечает за высокоточные сущности с устойчивым форматом:
- паспортные данные;
- телефоны;
- email;
- банковские карты;
- номера счетов;
- СНИЛС;
- ИНН;
- API-ключи и другие шаблонные объекты.

Преимущества regex-слоя:
- высокая точность на структурированных данных;
- быстрый запуск;
- предсказуемое поведение;
- отсутствие зависимости от модели.

### 2. NER Model

NER-модель решает задачу token classification и используется там, где одних правил недостаточно.

Обычно сюда относятся сущности, которые хуже описываются шаблонами:
- `PERSON_NAME`;
- `ADDRESS`;
- часть пограничных или контекстно-зависимых случаев.

### 3. Merge Layer

Merge-слой объединяет результаты regex и NER в единый итоговый список сущностей.

Он делает следующее:
- удаляет пересечения;
- разрешает конфликты;
- убирает дубликаты;
- сортирует сущности по позиции;
- сохраняет приоритет regex над NER.

---

## Поддерживаемые категории

Проект поддерживает следующие типы PII:
- `PASSPORT`
- `SNILS`
- `INN`
- `PHONE`
- `EMAIL`
- `CARD`
- `ACCOUNT`
- `ADDRESS`
- `API_KEY`
- `PERSON_NAME`

При необходимости список категорий можно расширять через правила, данные и конфигурацию модели.

---

## Структура проекта

```text
pii_ner_project/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ Makefile
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ answer/
├─ models/
├─ results/
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ types.py
│  │  ├─ io.py
│  │  ├─ logging_utils.py
│  │  └─ config.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ schemas.py
│  │  ├─ loader.py
│  │  └─ processor.py
│  ├─ regex/
│  │  ├─ __init__.py
│  │  ├─ patterns.py
│  │  └─ detector.py
│  ├─ ner/
│  │  ├─ __init__.py
│  │  ├─ tokenizer.py
│  │  ├─ model.py
│  │  ├─ dataset.py
│  │  ├─ trainer.py
│  │  └─ inference.py
│  ├─ merge/
│  │  ├─ __init__.py
│  │  ├─ merger.py
│  │  └─ span_utils.py
│  └─ metrics/
│     ├─ __init__.py
│     └─ strict_span_f1.py
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train_ner.py
│  ├─ predict_regex.py
│  ├─ predict_ner.py
│  ├─ predict_merged.py
│  ├─ evaluate.py
│  └─ run_tests.py
└─ tests/
   ├─ __init__.py
   ├─ fixtures.py
   ├─ test_regex.py
   ├─ test_ner.py
   ├─ test_merge.py
   ├─ test_data.py
   └─ test_metrics.py
```

Если у тебя часть скриптов или путей названа иначе, перед первым запуском синхронизируй README с реальной структурой репозитория.

Если у тебя часть скриптов или путей названа иначе, перед первым запуском синхронизируй README с реальной структурой репозитория.

Требования
Python 3.11

pip

GPU с CUDA — опционально, но желательно для обучения и быстрого инференса

Linux, macOS или Windows

Установка
Склонируй репозиторий:

bash
git clone <repo_url>
cd pii_ner_project
Создай виртуальное окружение:

bash
python3.11 -m venv venv
source venv/bin/activate
Для Windows:

bash
venv\Scripts\activate
Установи зависимости:

bash
pip install -r requirements.txt
Если проект использует .env, создай файл из шаблона:

bash
cp .env.example .env
После этого при необходимости отредактируй:

.env

configs/base.yaml

Что проверять сначала
Сначала нужно убедиться, что проект хотя бы базово согласован:

импорты не сломаны;

тестовая обвязка поднимается;

merge-логика работает;

CLI-скрипты запускаются.

Установи зависимости:

bash
pip install -r requirements.txt
Запусти все тесты:

bash
python scripts/run_tests.py --all
Если хочешь проверить только merge-логику:

bash
python scripts/run_tests.py --file tests/test_merge.py
Если хочешь посмотреть покрытие:

bash
python scripts/run_tests.py --all --coverage
Если run_tests.py в проекте пока нет, можно запускать напрямую через pytest:

bash
pytest tests/ -v
Только merge-тесты:

bash
pytest tests/test_merge.py -v
С покрытием:

bash
pytest --cov=src tests/
Если тесты проходят, значит структура импортов, merge-логика и базовые CLI-сценарии хотя бы минимально согласованы.

Как запускать пайплайн локально
Ниже минимальная рабочая последовательность.

1. Подготовка данных
Если у тебя уже есть:

data/processed/train.jsonl

data/processed/valid.jsonl

этот шаг можно пропустить.

Иначе:

bash
python scripts/prepare_data.py \
  --prepare-train data/raw/train.csv \
  --prepare-test data/raw/valid.csv \
  --output-dir data/processed
2. Обучение NER
bash
python scripts/train_ner.py \
  --config configs/base.yaml \
  --train-data data/processed/train.jsonl \
  --valid-data data/processed/valid.jsonl \
  --output-dir models/ner_checkpoint
3. Инференс regex-only
bash
python scripts/predict_regex.py \
  --test-data data/raw/private_test_dataset.csv \
  --output data/answer/predictions_regex.csv
4. Инференс NER-only
bash
python scripts/predict_ner.py \
  --model-path models/ner_checkpoint/model \
  --test-data data/raw/private_test_dataset.csv \
  --output data/answer/predictions_ner.jsonl \
  --device cuda
Если GPU нет:

bash
python scripts/predict_ner.py \
  --model-path models/ner_checkpoint/model \
  --test-data data/raw/private_test_dataset.csv \
  --output data/answer/predictions_ner.jsonl \
  --device cpu
5. Финальный merged inference
bash
python scripts/predict_merged.py \
  --test-data data/raw/private_test_dataset.csv \
  --model-path models/ner_checkpoint/model \
  --output data/answer/final_predictions.csv \
  --merge-strategy regex_priority \
  --device cuda
Если GPU нет:

bash
python scripts/predict_merged.py \
  --test-data data/raw/private_test_dataset.csv \
  --model-path models/ner_checkpoint/model \
  --output data/answer/final_predictions.csv \
  --merge-strategy regex_priority \
  --device cpu
Быстрый сценарий запуска
Если нужен самый короткий практический сценарий, делай так:

Установи зависимости.

Запусти тесты.

Подготовь данные, если они еще не подготовлены.

Обучи NER-модель.

Построй regex-only предсказания.

Построй NER-only предсказания.

Построй финальный merged-файл.

Проверь итоговый CSV.

Если есть gold-разметка, посчитай метрики.

Куда складываются артефакты
После запуска артефакты обычно лежат здесь.

Подготовленные данные:

data/processed/train.jsonl

data/processed/valid.jsonl

Чекпоинт модели:

models/ner_checkpoint/model/

models/ner_checkpoint/labeler.json

Предсказания:

regex-only: data/answer/predictions_regex.csv

ner-only: data/answer/predictions_ner.jsonl

финальный merged: data/answer/final_predictions.csv

Оценка:

results/evaluation.json

Итоговый файл для сабмита обычно:

data/answer/final_predictions.csv

Форматы данных
Обучающая выборка
Ожидается файл с текстом и списком сущностей.

Пример:

text
text	entities
Паспорт: 12 34 567890	[{"start": 9, "end": 20, "category": "PASSPORT"}]
Звоните: +7-999-123-45-67	[{"start": 9, "end": 24, "category": "PHONE"}]
Тестовая выборка
Тестовый файл содержит идентификатор и текст.

Пример:

text
id,text
1,"Личные данные..."
2,"Контактная информация..."
Формат предсказаний
Внутренне сущности должны быть представлены как:

python
[(start, end, category)]
Пример:

python
[(9, 20, "PASSPORT")]
Если для сабмита нужен CSV, он должен соответствовать формату, который ожидает организатор задачи.

Как проверить результат
Если есть gold-разметка, запускай:

bash
python scripts/evaluate.py \
  --gold data/gold.csv \
  --pred data/answer/final_predictions.csv \
  --output results/evaluation.json
Скрипт должен вернуть:

precision

recall

f1

Если gold-разметки нет, проверь хотя бы:

что итоговый CSV создался;

что файл не пустой;

что есть колонки id,prediction;

что prediction имеет вид [(10, 20, 'CATEGORY')] или [];

что между спанами нет пересечений;

что regex-сущности не были затерты NER-сущностями;

что сущности отсортированы по позиции в тексте.

Для быстрой ручной проверки:

открой первые строки CSV;

выбери 20–30 примеров;

сравни координаты с исходным текстом;

убедись, что координаты действительно попадают в нужные подстроки.

Метрика
Основная метрика качества:
Micro F1

Предсказание считается правильным только если одновременно совпали:

начало спана;

конец спана;

категория.

Частичное пересечение сущностей правильным ответом не считается.

Именно поэтому в проекте критично сохранять корректные offset mapping, не ломать границы сущностей и не допускать конфликтов после merge.

Конфигурация
Базовая конфигурация лежит в:

text
configs/base.yaml
Пример:

text
paths:
  data_dir: data/
  model_dir: models/

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  device: cuda

inference:
  batch_size: 64
  device: cuda
Перед запуском проверь:

пути к данным;

выходные директории;

device;

batch_size;

learning_rate;

число эпох;

параметры merge-логики, если они вынесены в конфиг.

Как работает merge
Merge-слой — это критичная часть проекта, потому что именно он формирует финальный ответ.

Типовая логика такая:

Забрать спаны от regex.

Забрать спаны от NER.

Нормализовать формат сущностей.

Удалить дубликаты.

Разрешить пересечения.

Оставить regex-предсказание при конфликте.

Отсортировать итоговый список по start.

Если у тебя в проекте несколько стратегий merge, удобно поддерживать параметр:

bash
--merge-strategy regex_priority
Это делает поведение пайплайна явным и воспроизводимым.

Локальный запуск по шагам
Ниже командами, без объяснений.

Подготовка данных:

bash
python scripts/prepare_data.py \
  --raw-train data/raw/train.csv \
  --raw-valid data/raw/valid.csv \
  --output-dir data/processed
Обучение:

bash
python scripts/train_ner.py \
  --config configs/base.yaml \
  --train-data data/processed/train.jsonl \
  --valid-data data/processed/valid.jsonl \
  --output-dir models/ner_checkpoint
Regex-only:

bash
python scripts/predict_regex.py \
  --test-data data/raw/private_test_dataset.csv \
  --output data/answer/predictions_regex.csv
NER-only:

bash
python scripts/predict_ner.py \
  --model-path models/ner_checkpoint/model \
  --test-data data/raw/private_test_dataset.csv \
  --output data/answer/predictions_ner.jsonl \
  --device cuda
Merged:

bash
python scripts/predict_merged.py \
  --test-data data/raw/private_test_dataset.csv \
  --model-path models/ner_checkpoint/model \
  --output data/answer/final_predictions.csv \
  --merge-strategy regex_priority \
  --device cuda
Оценка:

bash
python scripts/evaluate.py \
  --gold data/gold.csv \
  --pred data/answer/final_predictions.csv \
  --output results/evaluation.json
Google Colab
Если хочешь обучить модель и получить финальный файл в Colab, удобный сценарий такой.

1. Включи GPU
Открой:
Runtime -> Change runtime type -> GPU

2. Клонируй репозиторий
python
!git clone <URL_репозитория>
%cd <папка_репозитория>
3. Установи зависимости
python
!pip install -r requirements.txt
4. Подготовь данные, если нужно
python
!python scripts/prepare_data.py \
  --raw-train data/raw/train.csv \
  --raw-valid data/raw/valid.csv \
  --output-dir data/processed
5. Обучи модель
python
!python scripts/train_ner.py \
  --config configs/base.yaml \
  --train-data data/processed/train.jsonl \
  --valid-data data/processed/valid.jsonl \
  --output-dir models/ner_checkpoint
6. Получи финальный результат
python
!python scripts/predict_merged.py \
  --test-data data/raw/private_test_dataset.csv \
  --model-path models/ner_checkpoint/model \
  --output data/answer/final_predictions.csv \
  --merge-strategy regex_priority \
  --device cuda
7. Скачай итоговый CSV
python
from google.colab import files
files.download("data/answer/final_predictions.csv")
Полезные замечания
Перед первым полноценным запуском проверь, что:

реальные имена скриптов совпадают с README;

пути к train/valid/test данным существуют;

формат train соответствует ожидаемой схеме;

модель действительно сохраняется в models/ner_checkpoint;

merge-скрипт читает тот же формат, который выдает NER-скрипт.

Если в проекте уже есть старые команды вроде scripts/predict.py или scripts/train_ner.py в другом интерфейсе, лучше оставить один консистентный CLI и не дублировать несколько вариантов запуска без необходимости.

Ограничения
Текущая версия проекта может требовать дополнительных доработок в следующих направлениях:

улучшение постобработки multi-token сущностей;

потоковая обработка очень больших текстов;

расширение числа категорий;

confidence score для сущностей;

мультиязычная поддержка;

более строгая валидация входных и выходных форматов.

License
Proprietary.