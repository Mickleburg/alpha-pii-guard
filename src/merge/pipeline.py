"""PII detection pipeline combining regex and NER."""

from typing import List, Tuple, Optional
import time

from src.utils.logging_utils import get_logger
from src.merge.resolver import MergeResolver

logger = get_logger(__name__)

SpanTuple = Tuple[int, int, str]


class PIIDetectionPipeline:
    """End-to-end PII detection pipeline."""
    
    def __init__(
        self,
        regex_detector,
        ner_detector,
        merge_strategy: str = "regex_priority",
        timeout_seconds: float = 30.0
    ):
        """
        Initialize pipeline.
        
        Args:
            regex_detector: RegexPIIDetector instance
            ner_detector: NERInference instance
            merge_strategy: Strategy for merging predictions
            timeout_seconds: Timeout for processing
        """
        self.regex_detector = regex_detector
        self.ner_detector = ner_detector
        self.merger = MergeResolver(strategy=merge_strategy, remove_nested=True)
        self.timeout_seconds = timeout_seconds
    
    def predict(self, text: str) -> List[SpanTuple]:
        """
        Run full pipeline on a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, category) tuples
        """
        if not text or not text.strip():
            return []
        
        start_time = time.time()
        
        try:
            # Step 1: Regex detection
            regex_spans = self.regex_detector.predict(text)
            logger.debug(f"Regex detected {len(regex_spans)} spans")
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"Timeout reached after regex phase")
                return regex_spans
            
            # Step 2: NER detection
            ner_spans = self.ner_detector.predict(text)
            logger.debug(f"NER detected {len(ner_spans)} spans")
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"Timeout reached after NER phase")
                return self.merger.merge(regex_spans, [], text=text)
            
            # Step 3: Merge
            final_spans = self.merger.merge(regex_spans, ner_spans, text=text)
            logger.debug(f"Merged to {len(final_spans)} final spans")
            
            return final_spans
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Fallback: return regex only
            try:
                return self.regex_detector.predict(text)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return []
    
    def batch_predict(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[SpanTuple]]:
        """
        Run pipeline on multiple texts.
        
        Args:
            texts: List of texts
            show_progress: Whether to show progress logging
            
        Returns:
            List of span lists (parallel to input)
        """
        results = []
        
        for idx, text in enumerate(texts):
            if show_progress and (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(texts)}")
            
            spans = self.predict(text)
            results.append(spans)
        
        return results
    
    def predict_with_sources(self, text: str) -> dict:
        """
        Run pipeline and return predictions with source information.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with regex_spans, ner_spans, final_spans
        """
        regex_spans = self.regex_detector.predict(text)
        ner_spans = self.ner_detector.predict(text)
        final_spans = self.merger.merge(regex_spans, ner_spans, text=text)
        
        return {
            "regex_spans": regex_spans,
            "ner_spans": ner_spans,
            "final_spans": final_spans,
            "num_regex": len(regex_spans),
            "num_ner": len(ner_spans),
            "num_final": len(final_spans)
        }
