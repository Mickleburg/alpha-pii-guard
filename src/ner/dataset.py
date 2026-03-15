"""PyTorch dataset for NER training."""

from typing import List, Optional, Dict, Any
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.logging_utils import get_logger
from src.data.schemas import Document, Entity
from src.ner.labeling import BIOLabeler
from src.data.bio_converter import spans_to_bio, align_labels_to_tokens

logger = get_logger(__name__)


class NERDataset(Dataset):
    """PyTorch dataset for NER."""
    
    def __init__(
        self,
        documents: List[Document],
        tokenizer,
        labeler: BIOLabeler,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            documents: List of Document objects
            tokenizer: Tokenizer instance (e.g., from transformers)
            labeler: BIOLabeler instance
            max_length: Maximum sequence length
        """
        self.documents = documents
        self.tokenizer = tokenizer
        self.labeler = labeler
        self.max_length = max_length
        self.tokenized_data = []
        
        self._tokenize_documents()
    
    def _tokenize_documents(self):
        """Tokenize and prepare all documents."""
        skipped = 0
        
        for doc in self.documents:
            try:
                tokenized = self._tokenize_document(doc)
                if tokenized is not None:
                    self.tokenized_data.append(tokenized)
            except Exception as e:
                logger.warning(f"Failed to tokenize doc {doc.doc_id}: {e}")
                skipped += 1
        
        logger.info(
            f"Tokenized {len(self.tokenized_data)} documents "
            f"(skipped {skipped})"
        )
    
    def _tokenize_document(self, doc: Document) -> Optional[Dict[str, Any]]:
        """
        Tokenize single document.
        
        Returns:
            Dict with input_ids, attention_mask, token_type_ids, labels, offsets
        """
        text = doc.text
        entities = doc.entities
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors=None  # Don't convert to tensors yet
        )
        
        # Extract data
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))
        offset_mapping = encoded["offset_mapping"]
        
        # Create BIO labels
        num_tokens = len(input_ids)
        bio_labels = [0] * num_tokens  # Initialize with "O" (0)
        
        # Get span tuples
        entity_spans = [
            (e.start, e.end, e.category) for e in entities
        ]
        
        # Assign BIO labels to tokens
        for entity_start, entity_end, category in entity_spans:
            first_token = True
            
            for token_idx in range(num_tokens):
                token_start, token_end = offset_mapping[token_idx]
                
                # Skip special tokens (zero-length offsets)
                if token_start == token_end:
                    continue
                
                # Check overlap
                if token_start < entity_end and token_end > entity_start:
                    if first_token:
                        tag = f"B-{category}"
                        first_token = False
                    else:
                        tag = f"I-{category}"
                    
                    bio_labels[token_idx] = self.labeler.encode_tag(tag)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": bio_labels,
            "offset_mapping": offset_mapping,
            "text": text,
            "doc_id": doc.doc_id
        }
    
    def __len__(self) -> int:
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.tokenized_data[idx]
        
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "offset_mapping": item["offset_mapping"],
            "text": item["text"],
            "doc_id": item["doc_id"]
        }
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer,
        labeler: BIOLabeler,
        max_length: int = 512
    ) -> "NERDataset":
        """Load dataset from JSONL file."""
        documents = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                entities = [Entity(**e) for e in data.get("entities", [])]
                doc = Document(
                    text=data["text"],
                    entities=entities,
                    doc_id=data.get("doc_id")
                )
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {path}")
        return cls(documents, tokenizer, labeler, max_length)


def create_data_collator(labeler: BIOLabeler):
    """Create collate function for DataLoader."""

    def collate_fn(batch):
        """Collate batch of samples."""
        result = {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

        if all("token_type_ids" in item for item in batch):
            result["token_type_ids"] = torch.stack(
                [item["token_type_ids"] for item in batch]
            )

        return result

    return collate_fn

