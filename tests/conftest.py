"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import json

from src.regex.detector import create_detector
from src.ner.labeling import BIOLabeler
from src.merge.resolver import MergeResolver


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Иван Петров
    Телефон: +7 (495) 123-45-67
    Email: ivan.petrov@example.com
    Паспорт: 12 34 567890
    Адрес: г. Москва, ул. Ленина, д. 1
    """


@pytest.fixture
def sample_spans():
    """Sample spans for testing."""
    return [
        (10, 22, "ФИО"),
        (35, 55, "PHONE"),
        (57, 85, "EMAIL"),
        (95, 110, "PASSPORT"),
    ]


@pytest.fixture
def regex_detector():
    """Create regex detector."""
    return create_detector(use_context_rules=True)


@pytest.fixture
def bio_labeler():
    """Create BIO labeler."""
    categories = ["ФИО", "PHONE", "EMAIL", "PASSPORT", "ADDRESS"]
    return BIOLabeler(categories)


@pytest.fixture
def merge_resolver():
    """Create merge resolver."""
    return MergeResolver(strategy="regex_priority")


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "random_seed": 42,
        "paths": {
            "logs_dir": "logs",
            "models_dir": "models"
        },
        "regex": {
            "patterns_dir": "configs/patterns"
        },
        "ner": {
            "model_name": "DeepPavlov/rubert-base-cased",
            "tokenizer": {
                "max_length": 512
            }
        },
        "training": {
            "device": "cpu",
            "batch_size": 16,
            "eval_batch_size": 32,
            "epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 3,
            "logging_steps": 50,
            "early_stopping": {
                "patience": 3
            }
        }
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment."""
    # Suppress warnings during testing
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


@pytest.fixture
def example_document():
    """Create example document."""
    from src.data.schemas import Document, Entity
    
    text = "Иван Петров работает в Сбербанке. Его номер телефона +7 (495) 123-45-67."
    entities = [
        Entity(start=0, end=11, category="ФИО"),
        Entity(start=34, end=43, category="ORG"),
        Entity(start=65, end=85, category="PHONE"),
    ]
    
    return Document(text=text, entities=entities, doc_id="test_001")
