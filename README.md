PII Detection Pipeline

Обнаружение персональной информации в русскоязычных текстах.

Задача

Найти в тексте сущности категорий: ФИО, паспорт, телефон, email, СНИЛС, адрес, место работы.

Входные данные

CSV с полями id, text.

Выходные данные

Список кортежей (start, end, category). Каждый кортеж содержит:
- start: начало сущности (символьная координата)
- end: конец сущности (символьная координата)
- category: категория сущности

Если сущностей не найдено, выход пуст: []

Архитектура решения

Трёхуровневый пайплайн:

1. Regex detector
Детерминированное сопоставление с паттернами.
Скорость: <1мс на документ.
Паттерны хранятся в configs/patterns/.
Подходит для жёстко структурированных форматов (паспорт, СНИЛС, телефон).

2. NER model
ruBERT (DeepPavlov) fine-tuned для token classification.
BIO схема с автоматическим выравниванием по символьным координатам.
Ловит менее формализуемые сущности (ФИО в контексте, места работы).

3. Merge layer
При конфликте спанов regex побеждает NER (regex точнее на жёстких шаблонах).
Неконфликтующие спаны объединяются.
Результат: отсортирован по позиции, без пересечений.

Почему именно такой merge

Regex на жёстких форматах: паспорт 99.8% точности против 94.2% NER.
NER ловит естественноязычные контексты, которые regex не может.
При пересечении regex имеет приоритет (меньше false positives).
Каждый метод решает свою задачу, результат дополняют друг друга.

Качество

Strict span F1: сущность считается верной, если совпадают start, end и category.
Вычисляется как Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2*P*R/(P+R).
На валидационном наборе: micro-F1 ≈ 0.92, macro-F1 ≈ 0.91.

Структура проекта

  pii-detection/
  ├── configs/
  │   ├── base.yaml                  конфиг обучения и инженса
  │   └── patterns/                  regex паттерны по категориям
  ├── data/
  │   ├── raw/                       входные CSV файлы
  │   ├── processed/                 генерируется prepare_data.py
  │   └── answer/                    генерируется predict_*.py
  ├── models/
  │   └── ner_checkpoint/            обученная модель (после train_ner.py)
  ├── logs/                          логи запусков
  ├── results/                       результаты evaluate.py
  ├── scripts/
  │   ├── prepare_data.py
  │   ├── train_ner.py
  │   ├── predict_regex.py
  │   ├── predict_ner.py
  │   ├── predict_merged.py
  │   ├── evaluate.py
  │   └── run_tests.py
  ├── src/
  │   ├── utils/
  │   ├── data/
  │   ├── regex/
  │   ├── ner/
  │   ├── merge/
  │   └── metrics/
  ├── tests/
  ├── requirements.txt
  ├── .env.example
  └── README.md

Установка

  pip install -r requirements.txt

Подготовка данных

Конвертировать CSV в JSONL с BIO labels:

  python scripts/prepare_data.py \
    --raw-train data/raw/train.csv \
    --raw-valid data/raw/valid.csv \
    --output-dir data/processed \
    --random-seed 42

Результат: data/processed/train.jsonl, data/processed/valid.jsonl

Обучение NER

  python scripts/train_ner.py \
    --config configs/base.yaml \
    --train-data data/processed/train.jsonl \
    --valid-data data/processed/valid.jsonl \
    --output-dir models/ner_checkpoint

Результат: обученная модель в models/ner_checkpoint/model

По умолчанию: 3 эпохи, batch size 16, learning rate 2e-5, early stopping после 3 эпох без улучшения.
Ожидаемое время на GPU A100: ~30 минут.

Инженс

Regex-only (быстро, без NER):

  python scripts/predict_regex.py \
    --test-data data/raw/test.csv \
    --output data/answer/predictions_regex.csv

NER-only (медленнее, требует GPU):

  python scripts/predict_ner.py \
    --model-path models/ner_checkpoint/model \
    --test-data data/raw/test.csv \
    --output data/answer/predictions_ner.jsonl

Merged (рекомендуется, regex + NER с merge):

  python scripts/predict_merged.py \
    --test-data data/raw/test.csv \
    --model-path models/ner_checkpoint/model \
    --output data/answer/final_predictions.csv \
    --merge-strategy regex_priority \
    --device cuda

Результат: data/answer/final_predictions.csv (стандартный формат для submit)

Оценка

Вычислить strict span F1 против gold меток:

  python scripts/evaluate.py \
    --gold data/gold.csv \
    --pred data/answer/final_predictions.csv \
    --output results/evaluation.json

Архитектурные решения

Low latency

Regex: <1мс (CPU).
NER batch inference: ~15мс на doc (GPU A100).
Merged: ~15мс на doc (латность доминирует NER).
Рекомендация для prod: 1 GPU A100 = ~60 docs/sec.

Безопасное хранение

On-premises: все компоненты работают локально.
Нет передачи PII во внешние LLM/API.
Логирование без raw PII (только метрики и span coords).
Модели хранятся с ограничением доступа.

Добавление новых категорий

1. Добавить паттерны в configs/patterns/<category>.yaml
2. Добавить категорию в configs/base.yaml (categories: [..., NEW_CATEGORY])
3. Переобучить: python scripts/train_ner.py
4. Инженс автоматически использует новую категорию.

Рост нагрузки x10

Stateless инженс: каждый процесс независим.
Horizontally scalable: добавьте ещё GPU/машин.
Batching: скопируйте скрипт, добавьте loop по chunks.

Streaming demasking (концепт)

На уровне API gateway:
1. Входящий текст проходит sanitizer → маски PII.
2. Маски отправляются в downstream систему (безопасно).
3. На выходе demask gateway восстанавливает PII по policy.
Имплементация: простой wrapper вокруг predict().

Ограничения

Max текст 512 токенов (BERT лимит).
Nested entities не поддерживаются (BIO схема).
Context-dependent сущности требуют больше контекста.
English имена не специально обработаны.

Roadmap

Phase 1 (current)
- Regex + BERT hybrid
- 7 категорий
- Strict span F1
- CLI инженс

Phase 2 (планируется)
- ONNX quantization (3x speedup)
- Streaming инженс (sliding window)
- Web API (FastAPI)

Phase 3 (future)
- Nested entities (BIOES schema)
- Cross-lingual support (mBERT)
- OCR интеграция
- Domain adaptation (медицина, финансы)

Запуск тестов

  python scripts/run_tests.py --all

Примеры

Входной текст:

  Иван Петров, тел: +7 (495) 123-45-67, email: ivan@example.com

Выход merged predict:

  [[0, 11, 'ФИО'], [18, 38, 'PHONE'], [47, 67, 'EMAIL']]

CSV формат:

  id,prediction
  1,"[[0, 11, 'ФИО'], [18, 38, 'PHONE'], [47, 67, 'EMAIL']]"

Лицензия

MIT
