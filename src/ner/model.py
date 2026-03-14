"""BERT-based NER model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BertNER(nn.Module):
    """BERT-based token classification model for NER."""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_prob: float = 0.1
    ):
        """
        Initialize model.
        
        Args:
            model_name: Model name (e.g., "DeepPavlov/rubert-base-cased")
            num_labels: Number of BIO labels
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Load pre-trained BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification layer
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        logger.info(f"Initialized BertNER with {model_name}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            labels: Label IDs [batch_size, seq_len] (optional, for training)
            
        Returns:
            (logits, loss) where loss is None during inference
        """
        # BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout and classification
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            
            # Mask padding tokens
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.config.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fn.ignore_index).to(labels.device)
            )
            
            loss = loss_fn(active_logits, active_labels)
        
        return logits, loss
    
    def to_device(self, device: str) -> "BertNER":
        """Move model to device."""
        self.to(device)
        return self
    
    def train_mode(self) -> "BertNER":
        """Set to training mode."""
        self.train()
        return self
    
    def eval_mode(self) -> "BertNER":
        """Set to evaluation mode."""
        self.eval()
        return self


class NERModel:
    """Wrapper for BERT NER model with utilities."""
    
    def __init__(self, model_name: str, num_labels: int, device: str = "cuda"):
        """
        Initialize model wrapper.
        
        Args:
            model_name: Model name
            num_labels: Number of labels
            device: Device to use
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        
        self.model = BertNER(model_name, num_labels).to(device)
    
    def get_model(self) -> BertNER:
        """Get underlying model."""
        return self.model
    
    def save_pretrained(self, path: str) -> None:
        """Save model."""
        self.model.bert.save_pretrained(path)
        logger.info(f"Saved model to {path}")
    
    def load_pretrained(self, path: str) -> None:
        """Load model."""
        self.model.bert = AutoModel.from_pretrained(path)
        logger.info(f"Loaded model from {path}")
