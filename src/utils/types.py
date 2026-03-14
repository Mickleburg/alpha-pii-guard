"""Type definitions and models for the PII NER system."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Annotated

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Type alias for span tuple: (start_char, end_char, category)
SpanTuple = Tuple[int, int, str]


@dataclass
class Span:
    """Represents a single entity span in text."""
    
    start: int
    end: int
    category: str
    confidence: float = 1.0
    source: str = "unknown"  # 'regex' or 'ner'
    
    def to_tuple(self) -> SpanTuple:
        """Convert to tuple format (start, end, category)."""
        return (self.start, self.end, self.category)
    
    @classmethod
    def from_tuple(cls, span_tuple: SpanTuple, confidence: float = 1.0, source: str = "unknown") -> "Span":
        """Create Span from tuple format."""
        start, end, category = span_tuple
        return cls(start=start, end=end, category=category, confidence=confidence, source=source)
    
    def __repr__(self) -> str:
        return f"Span({self.start}, {self.end}, {self.category!r}, conf={self.confidence:.2f}, src={self.source!r})"
    
    def overlaps_with(self, other: "Span") -> bool:
        """Check if this span overlaps with another."""
        return not (self.end <= other.start or self.start >= other.end)
    
    def contains(self, other: "Span") -> bool:
        """Check if this span fully contains another."""
        return self.start <= other.start and self.end >= other.end
    
    def is_contained_by(self, other: "Span") -> bool:
        """Check if this span is fully contained by another."""
        return other.contains(self)


class EntityModel(BaseModel):
    """Pydantic model for a single entity in training data."""
    
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    category: Annotated[str, Field(min_length=1)]
    
    @field_validator("end")
    @classmethod
    def validate_end_greater_than_start(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v
    
    def to_span(self, confidence: float = 1.0, source: str = "unknown") -> Span:
        """Convert to Span object."""
        return Span(
            start=self.start,
            end=self.end,
            category=self.category,
            confidence=confidence,
            source=source
        )


class TrainSample(BaseModel):
    """Single training sample from TSV dataset."""
    
    text: str
    entities: List[EntityModel] = Field(default_factory=list)
    doc_id: Optional[str] = None
    
    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v


class DocumentPrediction(BaseModel):
    """Model for predictions on a single document."""
    
    doc_id: str
    text: str
    predictions: List[EntityModel]
    
    def to_span_tuples(self) -> List[SpanTuple]:
        """Convert predictions to span tuples."""
        return [pred.to_span().to_tuple() for pred in self.predictions]
    
    def get_spans(self, source: str = "unknown") -> List[Span]:
        """Get predictions as Span objects."""
        return [pred.to_span(source=source) for pred in self.predictions]


class BatchPrediction(BaseModel):
    """Model for batch predictions."""
    
    predictions: List[DocumentPrediction]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for easy serialization."""
        return [
            {
                "id": pred.doc_id,
                "text": pred.text,
                "predictions": pred.to_span_tuples()
            }
            for pred in self.predictions
        ]


class TokenizedSample(BaseModel):
    """Tokenized training sample with token-level labels."""
    
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]  # BIO tag indices
    offset_mapping: List[Tuple[int, int]]
    text: str
    doc_id: Optional[str] = None


class MetricsResult(BaseModel):
    """Model for evaluation metrics results."""
    
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)
    per_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"MetricsResult(P={self.precision:.3f}, R={self.recall:.3f}, F1={self.f1:.3f})"


class InferenceResult(BaseModel):
    """Result of inference on a single sample."""
    
    text: str
    spans: List[Span]
    processing_time_ms: float
    num_tokens: int
    model_used: str  # 'regex', 'ner', or 'merged'
    
    def to_tuple_list(self) -> List[SpanTuple]:
        """Get predictions as tuple list."""
        return [span.to_tuple() for span in self.spans]
