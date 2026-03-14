"""Inference utilities for NER model."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoConfig

from src.utils.logging_utils import get_logger
from src.ner.labeling import BIOLabeler, merge_tokens_to_spans
from src.ner.model import BertNER

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


class NERInference:
    """Inference pipeline for NER model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize inference.
        
        Args:
            model_path: Path to saved model directory
            device: Device to use
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.labeler = BIOLabeler.load(str(self.model_path / "labeler.json"))
        
        # Load model
        config = AutoConfig.from_pretrained(str(self.model_path))
        self.model = BertNER(
            str(self.model_path),
            config.num_labels
        ).to(device)
        self.model.eval()
        
        logger.info(f"Loaded NER model from {model_path}")
    
    def predict(self, text: str) -> List[SpanTuple]:
        """
        Predict entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, category) tuples
        """
        if not text or not text.strip():
            return []
        
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
            token_type_ids = token_type_ids.to(self.device)
            offset_mapping = encoded["offset_mapping"][0].tolist()
            
            # Forward pass
            logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        
        # Convert to spans
        spans = merge_tokens_to_spans(
            predictions,
            offset_mapping,
            self.labeler.id_to_tag,
            text=text
        )
        
        return spans
    
    def batch_predict(self, texts: List[str]) -> List[List[SpanTuple]]:
        """
        Predict entities in multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of span lists
        """
        return [self.predict(text) for text in texts]
    
    def predict_with_scores(self, text: str) -> List[Tuple[int, int, str, float]]:
        """
        Predict entities with confidence scores.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, category, confidence) tuples
        """
        if not text or not text.strip():
            return []
        
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
            token_type_ids = token_type_ids.to(self.device)
            offset_mapping = encoded["offset_mapping"][0].tolist()
            
            # Forward pass
            logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get predictions with scores
            probs = torch.softmax(logits, dim=-1)[0]
            predictions = torch.argmax(probs, dim=-1).cpu().tolist()
            scores = torch.max(probs, dim=-1)[0].cpu().tolist()
        
        # Convert to spans with scores
        spans = []
        current_entity_start = None
        current_entity_category = None
        current_entity_score = 0.0
        prev_token_end = None
        num_tokens = 0
        
        for token_idx, (pred_id, score, (token_start, token_end)) in enumerate(
            zip(predictions, scores, offset_mapping)
        ):
            tag = self.labeler.decode_tag(pred_id)
            
            if tag == "O":
                if current_entity_start is not None:
                    spans.append((
                        current_entity_start,
                        prev_token_end,
                        current_entity_category,
                        current_entity_score / num_tokens if num_tokens > 0 else 0.0
                    ))
                    current_entity_start = None
                    current_entity_category = None
                    num_tokens = 0
            
            elif tag.startswith("B-"):
                if current_entity_start is not None:
                    spans.append((
                        current_entity_start,
                        prev_token_end,
                        current_entity_category,
                        current_entity_score / num_tokens if num_tokens > 0 else 0.0
                    ))
                
                category = tag[2:]
                current_entity_start = token_start
                current_entity_category = category
                current_entity_score = score
                num_tokens = 1
            
            elif tag.startswith("I-"):
                category = tag[2:]
                
                if current_entity_start is None:
                    current_entity_start = token_start
                    current_entity_category = category
                    current_entity_score = score
                    num_tokens = 1
                elif current_entity_category != category:
                    spans.append((
                        current_entity_start,
                        prev_token_end,
                        current_entity_category,
                        current_entity_score / num_tokens if num_tokens > 0 else 0.0
                    ))
                    current_entity_start = token_start
                    current_entity_category = category
                    current_entity_score = score
                    num_tokens = 1
                else:
                    current_entity_score += score
                    num_tokens += 1
            
            prev_token_end = token_end
        
        if current_entity_start is not None:
            spans.append((
                current_entity_start,
                prev_token_end,
                current_entity_category,
                current_entity_score / num_tokens if num_tokens > 0 else 0.0
            ))
        
        return spans
