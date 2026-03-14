"""Tests for strict span F1 metrics."""

import pytest
import pandas as pd

from src.metrics.strict_span_f1 import (
    StrictSpanMatcher,
    precision_recall_f1,
    per_category_metrics,
    micro_f1_score,
    MetricsResult,
    evaluate_dataframe,
    format_metrics_report
)


class TestStrictSpanMatcher:
    """Test span matching logic."""
    
    def test_spans_equal_exact_match(self):
        """Test exact span match."""
        span1 = (0, 5, "PASSPORT")
        span2 = (0, 5, "PASSPORT")
        
        assert StrictSpanMatcher.spans_equal(span1, span2) is True
    
    def test_spans_equal_different_start(self):
        """Test spans with different start position."""
        span1 = (0, 5, "PASSPORT")
        span2 = (1, 5, "PASSPORT")
        
        assert StrictSpanMatcher.spans_equal(span1, span2) is False
    
    def test_spans_equal_different_end(self):
        """Test spans with different end position."""
        span1 = (0, 5, "PASSPORT")
        span2 = (0, 6, "PASSPORT")
        
        assert StrictSpanMatcher.spans_equal(span1, span2) is False
    
    def test_spans_equal_different_category(self):
        """Test spans with different category."""
        span1 = (0, 5, "PASSPORT")
        span2 = (0, 5, "PHONE")
        
        assert StrictSpanMatcher.spans_equal(span1, span2) is False
    
    def test_find_matches_empty_pred(self):
        """Test matching with empty predictions."""
        y_true = [(0, 5, "A")]
        y_pred = []
        
        matched_true, matched_pred = StrictSpanMatcher.find_matches(y_true, y_pred)
        
        assert matched_true == []
        assert matched_pred == []
    
    def test_find_matches_empty_true(self):
        """Test matching with empty ground truth."""
        y_true = []
        y_pred = [(0, 5, "A")]
        
        matched_true, matched_pred = StrictSpanMatcher.find_matches(y_true, y_pred)
        
        assert matched_true == []
        assert matched_pred == []
    
    def test_find_matches_exact_match(self):
        """Test finding exact matches."""
        y_true = [(0, 5, "A"), (10, 15, "B")]
        y_pred = [(0, 5, "A"), (10, 15, "B")]
        
        matched_true, matched_pred = StrictSpanMatcher.find_matches(y_true, y_pred)
        
        assert matched_true == [0, 1]
        assert matched_pred == [0, 1]
    
    def test_find_matches_partial_match(self):
        """Test with partial matches."""
        y_true = [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")]
        y_pred = [(0, 5, "A"), (10, 15, "B"), (30, 35, "D")]
        
        matched_true, matched_pred = StrictSpanMatcher.find_matches(y_true, y_pred)
        
        # Should match first two
        assert len(matched_true) == 2
        assert len(matched_pred) == 2
    
    def test_find_matches_no_matches(self):
        """Test with no matches."""
        y_true = [(0, 5, "A"), (10, 15, "B")]
        y_pred = [(1, 6, "A"), (11, 16, "B")]  # Shifted positions
        
        matched_true, matched_pred = StrictSpanMatcher.find_matches(y_true, y_pred)
        
        assert matched_true == []
        assert matched_pred == []


class TestPrecisionRecallF1:
    """Test precision, recall, F1 computation."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = [
            [(0, 5, "A"), (10, 15, "B")],
            [(5, 10, "C")]
        ]
        y_pred = [
            [(0, 5, "A"), (10, 15, "B")],
            [(5, 10, "C")]
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.tp == 3
        assert metrics.fp == 0
        assert metrics.fn == 0
    
    def test_no_predictions(self):
        """Test with no predictions."""
        y_true = [
            [(0, 5, "A"), (10, 15, "B")],
            [(5, 10, "C")]
        ]
        y_pred = [
            [],
            []
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 3
    
    def test_all_false_positives(self):
        """Test with all false positives."""
        y_true = [[], []]
        y_pred = [
            [(0, 5, "A"), (10, 15, "B")],
            [(5, 10, "C")]
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.tp == 0
        assert metrics.fp == 3
        assert metrics.fn == 0
    
    def test_partial_matches(self):
        """Test with partial matches."""
        y_true = [
            [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")]
        ]
        y_pred = [
            [(0, 5, "A"), (10, 15, "B")]  # Missing C
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.tp == 2
        assert metrics.fp == 0
        assert metrics.fn == 1
        assert metrics.precision == 1.0
        assert metrics.recall == 2/3
        expected_f1 = 2 * (1.0 * (2/3)) / (1.0 + 2/3)
        assert abs(metrics.f1 - expected_f1) < 1e-6
    
    def test_wrong_boundary_not_match(self):
        """Test that wrong boundary is not counted as match."""
        y_true = [
            [(0, 5, "A")]
        ]
        y_pred = [
            [(0, 6, "A")]  # Wrong end boundary
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.tp == 0
        assert metrics.fp == 1
        assert metrics.fn == 1
    
    def test_wrong_category_not_match(self):
        """Test that wrong category is not counted as match."""
        y_true = [
            [(0, 5, "A")]
        ]
        y_pred = [
            [(0, 5, "B")]  # Wrong category
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.tp == 0
        assert metrics.fp == 1
        assert metrics.fn == 1
    
    def test_mismatched_lengths(self):
        """Test error on mismatched lengths."""
        y_true = [[(0, 5, "A")]]
        y_pred = [[(0, 5, "A")], [(10, 15, "B")]]
        
        with pytest.raises(ValueError):
            precision_recall_f1(y_true, y_pred)
    
    def test_mixed_matches_and_misses(self):
        """Test with mix of matches and misses."""
        y_true = [
            [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")],
            [(5, 10, "D"), (15, 20, "E")]
        ]
        y_pred = [
            [(0, 5, "A"), (10, 15, "B"), (30, 35, "X")],  # 2 match, 1 fp, 1 fn
            [(5, 10, "D")]  # 1 match, 1 fn
        ]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.tp == 3
        assert metrics.fp == 1
        assert metrics.fn == 2
        assert metrics.precision == 3/4
        assert metrics.recall == 3/5


class TestPerCategoryMetrics:
    """Test per-category metric computation."""
    
    def test_single_category(self):
        """Test with single category."""
        y_true = [[(0, 5, "A"), (10, 15, "A")]]
        y_pred = [[(0, 5, "A")]]
        categories = ["A"]
        
        results = per_category_metrics(y_true, y_pred, categories=categories)
        
        assert "A" in results
        assert results["A"].tp == 1
        assert results["A"].fp == 0
        assert results["A"].fn == 1
    
    def test_multiple_categories(self):
        """Test with multiple categories."""
        y_true = [
            [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")]
        ]
        y_pred = [
            [(0, 5, "A"), (10, 15, "B"), (20, 25, "C")]
        ]
        categories = ["A", "B", "C"]
        
        results = per_category_metrics(y_true, y_pred, categories=categories)
        
        assert len(results) == 3
        for category in categories:
            assert results[category].tp == 1
            assert results[category].fp == 0
            assert results[category].fn == 0
            assert results[category].f1 == 1.0
    
    def test_category_extraction(self):
        """Test automatic category extraction."""
        y_true = [[(0, 5, "A"), (10, 15, "B")]]
        y_pred = [[(0, 5, "A"), (20, 25, "C")]]
        
        results = per_category_metrics(y_true, y_pred)
        
        # Should extract A, B, C
        assert "A" in results
        assert "B" in results
        assert "C" in results


class TestMicroF1Score:
    """Test micro-averaged F1 score."""
    
    def test_perfect_score(self):
        """Test perfect F1 score."""
        y_true = [[(0, 5, "A")]]
        y_pred = [[(0, 5, "A")]]
        
        f1 = micro_f1_score(y_true, y_pred)
        
        assert f1 == 1.0
    
    def test_zero_score(self):
        """Test zero F1 score."""
        y_true = [[(0, 5, "A")]]
        y_pred = [[(10, 15, "B")]]
        
        f1 = micro_f1_score(y_true, y_pred)
        
        assert f1 == 0.0
    
    def test_partial_score(self):
        """Test partial F1 score."""
        y_true = [[(0, 5, "A"), (10, 15, "B")]]
        y_pred = [[(0, 5, "A")]]
        
        f1 = micro_f1_score(y_true, y_pred)
        
        # P = 1/1 = 1.0, R = 1/2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 ≈ 0.667
        expected_f1 = 2 * (1.0 * 0.5) / (1.0 + 0.5)
        assert abs(f1 - expected_f1) < 1e-6


class TestMetricsResult:
    """Test MetricsResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MetricsResult(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            tp=9,
            fp=1,
            fn=2,
            support=11
        )
        
        result_dict = metrics.to_dict()
        
        assert result_dict["precision"] == 0.9
        assert result_dict["recall"] == 0.8
        assert result_dict["f1"] == 0.85
        assert result_dict["tp"] == 9
        assert result_dict["fp"] == 1
        assert result_dict["fn"] == 2
        assert result_dict["support"] == 11
    
    def test_repr(self):
        """Test string representation."""
        metrics = MetricsResult(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            tp=9,
            fp=1,
            fn=2
        )
        
        repr_str = repr(metrics)
        
        assert "0.9000" in repr_str or "0.9" in repr_str
        assert "0.8000" in repr_str or "0.8" in repr_str
        assert "0.85" in repr_str


class TestEvaluateDataframe:
    """Test dataframe evaluation."""
    
    def test_perfect_predictions_dataframe(self):
        """Test with perfect predictions in dataframe."""
        df_true = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["Sample 1", "Sample 2"],
            "predictions": [
                [(0, 5, "A"), (10, 15, "B")],
                [(5, 10, "C")]
            ]
        })
        
        df_pred = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["Sample 1", "Sample 2"],
            "predictions": [
                [(0, 5, "A"), (10, 15, "B")],
                [(5, 10, "C")]
            ]
        })
        
        result = evaluate_dataframe(df_true, df_pred)
        
        assert result["overall"].f1 == 1.0
        assert result["num_documents"] == 2
        assert result["num_true_entities"] == 3
        assert result["num_pred_entities"] == 3
    
    def test_mismatched_lengths(self):
        """Test error on mismatched dataframe lengths."""
        df_true = pd.DataFrame({
            "id": ["1"],
            "text": ["Sample 1"],
            "predictions": [[(0, 5, "A")]]
        })
        
        df_pred = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["Sample 1", "Sample 2"],
            "predictions": [[(0, 5, "A")], [(5, 10, "B")]]
        })
        
        with pytest.raises(ValueError):
            evaluate_dataframe(df_true, df_pred)


class TestFormatMetricsReport:
    """Test metrics report formatting."""
    
    def test_format_overall_only(self):
        """Test formatting with overall metrics only."""
        metrics = MetricsResult(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            tp=9,
            fp=1,
            fn=2,
            support=11
        )
        
        report = format_metrics_report(metrics)
        
        assert "STRICT SPAN + CATEGORY F1 METRICS" in report
        assert "OVERALL METRICS:" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1-Score" in report
    
    def test_format_with_per_category(self):
        """Test formatting with per-category metrics."""
        metrics = MetricsResult(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            tp=9,
            fp=1,
            fn=2,
            support=11
        )
        
        per_category = {
            "A": MetricsResult(precision=0.95, recall=0.90, f1=0.925, tp=9, fp=0, fn=1, support=10),
            "B": MetricsResult(precision=0.5, recall=0.5, f1=0.5, tp=0, fp=1, fn=1, support=1)
        }
        
        report = format_metrics_report(metrics, per_category=per_category)
        
        assert "PER-CATEGORY METRICS:" in report
        assert "A" in report
        assert "B" in report


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_empty_documents(self):
        """Test with empty documents."""
        y_true = [[], []]
        y_pred = [[], []]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        # Zero division case
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 0
    
    def test_duplicate_spans_in_pred(self):
        """Test with duplicate spans in predictions."""
        y_true = [[(0, 5, "A")]]
        y_pred = [[(0, 5, "A"), (0, 5, "A")]]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        # Only one should match
        assert metrics.tp == 1
        assert metrics.fp == 1
        assert metrics.fn == 0
    
    def test_overlapping_different_categories(self):
        """Test overlapping spans with different categories."""
        y_true = [[(0, 10, "A")]]
        y_pred = [[(0, 5, "B"), (5, 10, "C")]]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        # No match (different categories and boundaries)
        assert metrics.tp == 0
        assert metrics.fp == 2
        assert metrics.fn == 1
    
    def test_many_documents(self):
        """Test with many documents."""
        # Create 100 documents
        y_true = [[(i, i+5, "A")] for i in range(0, 500, 5)]
        y_pred = [[(i, i+5, "A")] for i in range(0, 500, 5)]
        
        metrics = precision_recall_f1(y_true, y_pred)
        
        assert metrics.tp == 100
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.f1 == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
