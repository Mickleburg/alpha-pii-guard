import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm
import os


from src.labels import LABEL2ID, ID2LABEL, LABELS
from src.prepare_data import spans_to_bio_tags


# ============================================================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ============================================================================


# Опции моделей для русского языка NER
MODEL_OPTIONS = {
    "tiny": "cointegrated/rubert-tiny2",           # ~6M params, ~100MB
    "base": "cointegrated/rubert-base-cased",      # ~110M params, ~440MB
    "large": "cointegrated/rubert-large-cased",    # ~360M params, ~1.4GB
    "deeppavlov": "DeepPavlov/rubert-base-cased",  # ~110M params, производственный
}


# ============================================================================
# NER ДАТАСЕТ
# ============================================================================


class NERDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        spans: List[List[Tuple[int, int, str]]],
        tokenizer,
        max_len: int = 512
    ):
        """
        Датасет для токен-классификации NER.
        
        Args:
            texts: Список текстов
            spans: Список списков spans (start, end, label)
            tokenizer: Токенайзер из transformers
            max_len: Максимальная длина последовательности
        """
        self.texts = texts
        self.spans = spans
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = []
        self._prepare()
    
    def _prepare(self):
        """Подготавливает данные к обучению."""
        for text, span_list in zip(self.texts, self.spans):
            # Конвертируем spans в BIO-теги на уровне символов
            bio_tags = spans_to_bio_tags(text, span_list)
            
            # Токенизируем с сохранением информации об offsets
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_offsets_mapping=True,
            )
            
            # Конвертируем символьные BIO-теги в токенные
            labels = []
            for i, offset in enumerate(encoding["offset_mapping"]):
                start_char, end_char = offset
                
                # Токены для специальных символов и padding
                if start_char == end_char:
                    labels.append(-100)  # Игнорируем при вычислении loss
                else:
                    # Берём тег для первого символа токена
                    if start_char < len(bio_tags):
                        tag = bio_tags[start_char]
                        label_id = LABEL2ID.get(tag, LABEL2ID.get("O", 0))
                    else:
                        label_id = LABEL2ID.get("O", 0)
                    
                    labels.append(label_id)
            
            encoding["labels"] = labels
            del encoding["offset_mapping"]
            self.encodings.append(encoding)
    
    def __len__(self) -> int:
        return len(self.encodings)
    
    def __getitem__(self, idx: int) -> dict:
        enc = self.encodings[idx]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }



# ============================================================================
# NER МОДЕЛЬ
# ============================================================================


class NERModel:
    def __init__(
        self,
        model_name: str = "tiny",  # "tiny", "base", "large", "deeppavlov" (по умолчанию tiny!)
        output_dir: str = "ner_model",
        device: str = None
    ):
        """
        Инициализирует NER модель.
        
        Args:
            model_name: Название модели из MODEL_OPTIONS или полный путь
            output_dir: Директория для сохранения модели
            device: Устройство вычисления ("cuda", "cpu")
        """
        self.output_dir = output_dir
        
        # Резолвим название модели
        if model_name in MODEL_OPTIONS:
            self.model_name = MODEL_OPTIONS[model_name]
        else:
            self.model_name = model_name
        
        # Устанавливаем device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Using model: {self.model_name}")
        
        # Загружаем токенайзер и модель
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # ✅ ИСПРАВЛЕНИЕ 1: Правильное число лабелов
        # LABEL2ID содержит: "O" + B- и I- версии для каждого LABELS
        # Всего: 1 + 30*2 = 61
        num_labels = len(LABEL2ID)
        print(f"Number of labels: {num_labels}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels  # ✅ ИСПРАВЛЕНО
        )
        
        # ✅ ИСПРАВЛЕНИЕ 2: Убираем .to(device) после multi-GPU инициализации
        # Это вызывает проблемы с DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            # Для transformers лучше использовать device_map вместо DataParallel
            # Но для совместимости оставим просто на первый GPU
            self.model = self.model.to(self.device)
        else:
            self.model.to(self.device)
    
    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_len: int = 512,
        save_steps: int = 100
    ) -> dict:
        """
        Обучает NER модель.
        
        Args:
            train_df: DataFrame с колонками ['text', 'target']
            valid_df: Валидационный DataFrame (опционально)
            epochs: Число эпох обучения
            batch_size: Размер batch'а
            learning_rate: Learning rate для оптимизатора
            max_len: Максимальная длина последовательности
            save_steps: Каждые сколько шагов сохранять checkpoint
        
        Returns:
            История обучения
        """
        # Подготавливаем датасеты
        train_dataset = NERDataset(
            train_df["text"].tolist(),
            train_df["target"].tolist(),
            self.tokenizer,
            max_len=max_len,
        )
        
        # ✅ ИСПРАВЛЕНИЕ 3: Правильные TrainingArguments
        # - eval_strategy="no" вместо save_strategy по умолчанию
        # - gradient_accumulation_steps для экономии памяти
        # - fp16 только если есть GPU
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=50,
            seed=42,
            eval_strategy="no",  # ✅ ИСПРАВЛЕНО: Не требуем eval датасета
            save_strategy="steps",  # Сохраняем по шагам
            gradient_accumulation_steps=2,  # ✅ ДОБАВЛЕНО: Для экономии памяти
            fp16=torch.cuda.is_available(),  # Mixed precision если доступен GPU
            dataloader_pin_memory=True,
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",  # ✅ Экономия памяти
        )
        
        # ✅ ИСПРАВЛЕНИЕ 4: Убираем eval_dataset=None
        # Создаём Trainer без eval_dataset (поскольку eval_strategy="no")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset убран, т.к. eval_strategy="no"
        )
        
        # Обучаем
        trainer.train()
        
        # Сохраняем
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
        return trainer.state.log_history
    
    def predict_text(self, text: str, max_len: int = 512) -> List[Tuple[int, int, str]]:
        """
        Предсказывает spans для одного текста.
        
        Args:
            text: Входной текст
            max_len: Максимальная длина последовательности
        
        Returns:
            Список (start, end, label) spans
        """
        # ✅ ИСПРАВЛЕНИЕ 5: Явно указываем return_offsets_mapping=True
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,  # ✅ ЯВНО УКАЗАНО
            return_tensors="pt",
        )
        
        offset_mapping = encoding.pop("offset_mapping")[0].numpy()
        
        # Предсказываем
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in encoding.items()})
            logits = outputs.logits[0].cpu().numpy()
        
        # Получаем pred_labels
        pred_ids = np.argmax(logits, axis=1)
        
        # Конвертируем pred_ids в spans
        spans = []
        current_span = None
        current_label = None
        
        for i, (pred_id, offset) in enumerate(zip(pred_ids, offset_mapping)):
            start_char, end_char = offset
            
            # Пропускаем специальные токены
            if start_char == end_char or pred_id == 0:  # 0 = "O" tag
                if current_span is not None:
                    spans.append((current_span[0], current_span[1], current_label))
                current_span = None
                current_label = None
            else:
                # Получаем label из pred_id
                label_tag = ID2LABEL.get(pred_id, "O")
                
                # Убираем B-/I- префиксы
                label = label_tag.replace("B-", "").replace("I-", "")
                
                # Проверяем, начинается ли новый span (B- tag)
                if label_tag.startswith("B-") or current_label != label:
                    if current_span is not None:
                        spans.append((current_span[0], current_span[1], current_label))
                    current_span = (start_char, end_char)
                    current_label = label
                else:
                    # Продолжаем текущий span
                    if current_span is not None:
                        current_span = (current_span[0], end_char)
        
        # Добавляем последний span
        if current_span is not None:
            spans.append((current_span[0], current_span[1], current_label))
        
        return sorted(spans, key=lambda x: x[0])
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Предсказывает spans для batch'а текстов.
        
        Args:
            texts: Список текстов
            batch_size: Размер batch'а для ускорения обработки
        
        Returns:
            Список списков spans
        """
        predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="NER prediction"):
            batch_texts = texts[i:i+batch_size]
            for text in batch_texts:
                pred = self.predict_text(text)
                predictions.append(pred)
        
        return predictions
    
    def load(self):
        """Загружает сохранённую модель."""
        if os.path.exists(self.output_dir):
            self.model = AutoModelForTokenClassification.from_pretrained(self.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            self.model.to(self.device)
            print(f"Model loaded from {self.output_dir}")
        else:
            raise FileNotFoundError(f"Model directory {self.output_dir} not found")