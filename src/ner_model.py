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
from src.utils import bio_to_spans

class NERDataset(Dataset):
    def __init__(self, texts: List[str], spans: List[List[Tuple[int, int, str]]], tokenizer, max_len=512):
        self.texts = texts
        self.spans = spans
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = []
        self._prepare()
    
    def _prepare(self):
        for text, span_list in zip(self.texts, self.spans):
            bio_tags = spans_to_bio_tags(text, span_list)
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_offsets_mapping=True,
            )
            
            labels = []
            for i, offset in enumerate(encoding["offset_mapping"]):
                if offset[0] == offset[1]:
                    # Специальный токен
                    labels.append(-100)
                else:
                    char_idx = offset[0]
                    if char_idx < len(bio_tags):
                        tag = bio_tags[char_idx]
                        labels.append(LABEL2ID.get(tag, LABEL2ID.get("O", 0)))
                    else:
                        labels.append(LABEL2ID.get("O", 0))
            
            encoding["labels"] = labels
            del encoding["offset_mapping"]
            self.encodings.append(encoding)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        enc = self.encodings[idx]
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(enc["labels"]),
        }

class NERModel:
    def __init__(self, model_name="cointegrated/rubert-tiny2", output_dir="ner_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Всего меток: O + B-label и I-label для каждого из 30 лейблов
        num_labels = len(LABEL2ID)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    
    def train(self, train_df, epochs=3, batch_size=8, max_len=512):
        """Обучаем модель на всём train датасете (без валидации)."""
        train_dataset = NERDataset(
            train_df["text"].tolist(),
            train_df["target"].tolist(),
            self.tokenizer,
            max_len=max_len,
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
            seed=42,
            learning_rate=2e-5,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
    
    def predict_text(self, text: str, max_len=512) -> List[Tuple[int, int, str]]:
        """Предсказываем на одном тексте. Возвращаем список кортежей (start, end, label)."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        
        offset_mapping = encoding.pop("offset_mapping")[0].numpy()
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits[0].cpu().numpy()
        
        pred_labels = np.argmax(logits, axis=1)
        
        # Преобразуем ID в BIO-теги
        bio_tags = []
        for pred_id in pred_labels:
            tag = ID2LABEL.get(pred_id, "O")
            bio_tags.append(tag)
        
        # Преобразуем BIO-теги в spans
        spans = bio_to_spans(text, bio_tags)
        return sorted(spans, key=lambda x: x[0])
    
    def predict_batch(self, texts: List[str]) -> List[List[Tuple[int, int, str]]]:
        """Предсказываем на батче текстов. Возвращаем список списков кортежей."""
        predictions = []
        for text in tqdm(texts, desc="NER prediction"):
            pred = self.predict_text(text)
            predictions.append(pred)
        return predictions
    
    def load(self):
        """Загружаем модель с диска."""
        if os.path.exists(self.output_dir):
            self.model = AutoModelForTokenClassification.from_pretrained(self.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        else:
            raise FileNotFoundError(f"Model not found at {self.output_dir}")