"""
Tests for unified metrics module.
"""

import pytest
from ml.entities import Entity
from ml.eval.metrics_rules_ner import (
    to_normalized_set,
    compute_counts,
    precision_recall_f1,
    compute_micro_metrics,
    compute_per_label_metrics,
)


def test_exact_tp():
    """Exact match counts as TP."""
    pred = {(0, 5, "Email")}
    gold = {(0, 5, "Email")}
    
    tp, fp, fn = compute_counts(pred, gold)
    assert tp == 1
    assert fp == 0
    assert fn == 0


def test_wrong_label_fp_fn():
    """Wrong label counts as FP + FN."""
    pred = {(0, 5, "Email")}
    gold = {(0, 5, "ФИО")}
    
    tp, fp, fn = compute_counts(pred, gold)
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_wrong_span_fp_fn():
    """Wrong span counts as FP + FN."""
    pred = {(0, 5, "Email")}
    gold = {(0, 10, "Email")}
    
    tp, fp, fn = compute_counts(pred, gold)
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_bio_labels_normalized():
    """BIO labels normalized before comparison."""
    pred_tuples = [(0, 5, "B-Email")]
    gold_tuples = [(0, 5, "Email")]
    
    pred = to_normalized_set(pred_tuples)
    gold = to_normalized_set(gold_tuples)
    
    tp, fp, fn = compute_counts(pred, gold)
    assert tp == 1


def test_micro_precision_recall_f1():
    """Micro metrics computed correctly."""
    predictions = [
        [(0, 5, "Email"), (10, 15, "ФИО")],
        [(0, 10, "Номер телефона")],
    ]
    references = [
        [(0, 5, "Email"), (10, 15, "ФИО")],
        [(0, 10, "Номер телефона"), (20, 30, "Паспортные данные")],
    ]
    
    metrics = compute_micro_metrics(predictions, references)
    assert metrics['tp'] == 3
    assert metrics['fp'] == 0
    assert metrics['fn'] == 1
    assert metrics['precision'] == 1.0
    assert 0.7 < metrics['recall'] < 0.8
    assert 0.8 < metrics['f1'] < 0.9


def test_per_label_metrics():
    """Per-label metrics computed correctly."""
    predictions = [
        [(0, 5, "Email")],
        [(0, 10, "ФИО"), (20, 30, "Email")],
    ]
    references = [
        [(0, 5, "Email"), (10, 15, "Номер телефона")],
        [(0, 10, "ФИО")],
    ]
    
    metrics = compute_per_label_metrics(predictions, references)
    
    assert "Email" in metrics
    assert metrics["Email"]["tp"] == 1
    assert metrics["Email"]["fp"] == 1
    assert metrics["Email"]["fn"] == 0
    
    assert "ФИО" in metrics
    assert metrics["ФИО"]["tp"] == 1
    
    assert "Номер телефона" in metrics
    assert metrics["Номер телефона"]["fn"] == 1


def test_empty_predictions():
    """Empty predictions handled correctly."""
    predictions = [[]]
    references = [[(0, 5, "Email")]]
    
    metrics = compute_micro_metrics(predictions, references)
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['f1'] == 0.0


def test_empty_references():
    """Empty references handled correctly."""
    predictions = [[(0, 5, "Email")]]
    references = [[]]
    
    metrics = compute_micro_metrics(predictions, references)
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['f1'] == 0.0


def test_entity_input():
    """Supports Entity object input."""
    pred_entities = [[Entity(0, 5, "Email", "regex")]]
    gold_tuples = [[(0, 5, "Email")]]
    
    metrics = compute_micro_metrics(pred_entities, gold_tuples)
    assert metrics['tp'] == 1
    assert metrics['fp'] == 0
    assert metrics['fn'] == 0


def test_mixed_tuple_entity_input():
    """Supports mixed tuple and Entity input."""
    pred = [[(0, 5, "Email")]]
    gold = [[Entity(0, 5, "Email", "gold")]]
    
    metrics = compute_micro_metrics(pred, gold)
    assert metrics['tp'] == 1
