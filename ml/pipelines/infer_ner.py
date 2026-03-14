# File: ml/pipelines/infer_ner.py
import torch
from typing import List, Tuple, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

class NERModel:
    """NER inference model for PII detection."""
    
    def __init__(self, model_dir: str = "ml/models/ner/"):
        """Initialize model and tokenizer.
        
        Args:
            model_dir: Path to saved model directory
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading NER model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        print(f"Model loaded on {self.device}")
    
    def _decode_bio_to_spans(
        self,
        tokens: List[str],
        labels: List[str],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str]]:
        """Convert BIO labels back to character-level spans.
        
        Args:
            tokens: List of tokens
            labels: List of BIO labels
            offset_mapping: Character offsets for each token
        
        Returns:
            List of (start_char, end_char, category) tuples
        """
        spans = []
        current_entity = None
        current_start = None
        current_end = None
        
        for token, label, (char_start, char_end) in zip(tokens, labels, offset_mapping):
            # Skip special tokens
            if char_start == char_end:
                if current_entity:
                    spans.append((current_start, current_end, current_entity))
                    current_entity = None
                continue
            
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    spans.append((current_start, current_end, current_entity))
                
                # Start new entity
                current_entity = label[2:]
                current_start = char_start
                current_end = char_end
            
            elif label.startswith('I-'):
                category = label[2:]
                if current_entity == category:
                    # Extend current entity
                    current_end = char_end
                else:
                    # Mismatched I- tag, start new entity
                    if current_entity:
                        spans.append((current_start, current_end, current_entity))
                    current_entity = category
                    current_start = char_start
                    current_end = char_end
            
            else:  # 'O' label
                if current_entity:
                    spans.append((current_start, current_end, current_entity))
                    current_entity = None
        
        # Save final entity if exists
        if current_entity:
            spans.append((current_start, current_end, current_entity))
        
        return spans
    
    def predict(self, text: str) -> List[Tuple[int, int, str]]:
        """Predict entities in text.
        
        Args:
            text: Input text
        
        Returns:
            List of (start_char, end_char, category) tuples
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Handle long texts with sliding window
        max_length = 510  # Reserve space for special tokens
        stride = 64
        
        if len(text) <= max_length:
            return self._predict_single(text)
        else:
            return self._predict_sliding_window(text, max_length, stride)
    
    def _predict_single(self, text: str) -> List[Tuple[int, int, str]]:
        """Predict entities for a single text segment."""
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        offset_mapping = encoding.pop('offset_mapping')[0].tolist()
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        # Convert to labels
        predicted_labels = [self.id2label[pred.item()] for pred in predictions]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Decode to spans
        spans = self._decode_bio_to_spans(tokens, predicted_labels, offset_mapping)
        
        return spans
    
    def _predict_sliding_window(
        self,
        text: str,
        max_length: int,
        stride: int
    ) -> List[Tuple[int, int, str]]:
        """Predict entities using sliding window for long texts."""
        all_spans = []
        text_length = len(text)
        
        for start_idx in range(0, text_length, max_length - stride):
            end_idx = min(start_idx + max_length, text_length)
            segment = text[start_idx:end_idx]
            
            segment_spans = self._predict_single(segment)
            
            # Adjust offsets to original text
            adjusted_spans = [
                (start + start_idx, end + start_idx, cat)
                for start, end, cat in segment_spans
            ]
            
            all_spans.extend(adjusted_spans)
            
            if end_idx >= text_length:
                break
        
        # Deduplicate overlapping spans (keep first occurrence)
        seen_spans = set()
        unique_spans = []
        
        for span in all_spans:
            span_key = (span[0], span[1], span[2])
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_spans.append(span)
        
        return sorted(unique_spans, key=lambda x: x[0])
    
    def predict_batch(self, texts: List[str]) -> List[List[Tuple[int, int, str]]]:
        """Predict entities for multiple texts in batch.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of predictions, one per input text
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [(idx, text) for idx, text in enumerate(texts) if text and len(text.strip()) > 0]
        
        if not non_empty_texts:
            return [[] for _ in texts]
        
        indices, valid_texts = zip(*non_empty_texts)
        
        # Tokenize batch
        encodings = self.tokenizer(
            list(valid_texts),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        offset_mappings = encodings.pop('offset_mapping')
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode each sequence
        batch_spans = []
        for idx, (preds, offset_map) in enumerate(zip(predictions, offset_mappings)):
            predicted_labels = [self.id2label[pred.item()] for pred in preds]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[idx])
            
            spans = self._decode_bio_to_spans(tokens, predicted_labels, offset_map.tolist())
            batch_spans.append(spans)
        
        # Reconstruct full results with empty lists for empty inputs
        results = [[] for _ in texts]
        for result_idx, original_idx in enumerate(indices):
            results[original_idx] = batch_spans[result_idx]
        
        return results
