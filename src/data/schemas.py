"""Pydantic schemas for data structures."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator


class Entity(BaseModel):
    """Single entity/annotation."""
    
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    category: str = Field(min_length=1)
    
    @field_validator("end")
    @classmethod
    def validate_end_gt_start(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be > start")
        return v


class Document(BaseModel):
    """Single document with text and annotations."""
    
    text: str = Field(min_length=1)
    entities: List[Entity] = Field(default_factory=list)
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v


class Batch(BaseModel):
    """Batch of documents."""
    
    documents: List[Document]
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class TokenizedDocument:
    """Tokenized document with token-level labels."""
    
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]  # BIO tag IDs
    offset_mapping: List[tuple[int, int]]
    text: str
    doc_id: Optional[str] = None
    original_entities: Optional[List[Entity]] = None
    
    def __len__(self) -> int:
        return len(self.input_ids)


class BioBatch(BaseModel):
    """Batch of tokenized documents."""
    
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    token_type_ids: List[List[int]]
    labels: List[List[int]]
    offset_mapping: List[List[tuple[int, int]]]
    texts: List[str]
    doc_ids: List[str]
    original_entities: List[List[Entity]] = Field(default_factory=list)


class PredictionBatch(BaseModel):
    """Batch of predictions."""
    
    doc_ids: List[str]
    texts: List[str]
    predicted_labels: List[List[int]]  # BIO label IDs
    offset_mapping: List[List[tuple[int, int]]]
    scores: Optional[List[List[float]]] = None
    
    
class Span(BaseModel):
    """Detected span in text."""
    
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    category: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="unknown")  # 'regex', 'ner', 'merged'


class TextWithSpans(BaseModel):
    """Text with detected spans."""
    
    text: str
    spans: List[Span]
    doc_id: Optional[str] = None
    processing_time_ms: Optional[float] = None


@dataclass
class BioBatch_Collated:
    """Collated batch ready for model input (numpy/torch arrays)."""
    
    input_ids: Any  # torch.Tensor
    attention_mask: Any
    token_type_ids: Any
    labels: Any
    offset_mapping: List[List[tuple[int, int]]]
    texts: List[str]
    doc_ids: List[str]
