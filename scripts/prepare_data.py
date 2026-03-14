"""Prepare and preprocess raw data."""

import json
from pathlib import Path
from typing import List

import argparse

from src.utils.config import load_config, get_labels
from src.utils.logging_utils import get_logger, setup_logging
from src.data.readers import load_train_dataset, load_test_dataset, save_jsonl_documents
from src.data.preprocessing import (
    preprocess_documents,
    train_valid_split,
    calculate_dataset_statistics,
    print_dataset_statistics
)
from src.data.bio_converter import get_bio_tag_scheme

logger = get_logger(__name__)


def prepare_train_data(
    config_path: str = "configs/base.yaml",
    output_dir: str = "data/processed"
) -> None:
    """
    Prepare training data:
    - Load raw train dataset
    - Validate and preprocess
    - Split into train/valid
    - Save as JSONL
    - Save label mappings
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for processed data
    """
    # Load config
    config = load_config(config_path)
    setup_logging(log_dir=config["paths"]["logs_dir"])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PREPARING TRAINING DATA")
    logger.info("="*60)
    
    # Get paths
    train_path = Path(config["paths"]["train_data"])
    logger.info(f"Loading raw train data from: {train_path}")
    
    # Load raw data
    raw_documents = load_train_dataset(train_path, validate=True)
    logger.info(f"Loaded {len(raw_documents)} raw documents")
    
    # Preprocess
    logger.info("Preprocessing documents...")
    processed_documents, skipped = preprocess_documents(
        raw_documents,
        no_text_mutation=True,
        validate=True
    )
    logger.info(f"Preprocessed: {len(processed_documents)} documents, skipped: {skipped}")
    
    if not processed_documents:
        logger.error("No documents after preprocessing!")
        return
    
    # Calculate statistics before split
    logger.info("Calculating dataset statistics...")
    stats = calculate_dataset_statistics(processed_documents)
    print_dataset_statistics(stats)
    
    # Train/valid split
    train_ratio = config.get("data", {}).get("train_ratio", 0.8)
    val_ratio = config.get("data", {}).get("val_ratio", 0.1)
    
    logger.info(f"Splitting with train_ratio={train_ratio}...")
    train_docs, valid_docs = train_valid_split(
        processed_documents,
        train_ratio=train_ratio,
        seed=config["random_seed"]
    )
    
    # Save train data
    train_path = output_dir / "train.jsonl"
    save_jsonl_documents(train_docs, train_path)
    logger.info(f"Saved {len(train_docs)} training documents to {train_path}")
    
    # Save valid data
    valid_path = output_dir / "valid.jsonl"
    save_jsonl_documents(valid_docs, valid_path)
    logger.info(f"Saved {len(valid_docs)} validation documents to {valid_path}")
    
    # Save train statistics
    train_stats = calculate_dataset_statistics(train_docs)
    valid_stats = calculate_dataset_statistics(valid_docs)
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": stats,
            "train": train_stats,
            "valid": valid_stats
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to {stats_path}")
    
    # Save label mappings
    logger.info("Creating label mappings...")
    categories = get_labels(config)
    bio_tags, tag_to_id, id_to_tag = get_bio_tag_scheme(categories)
    
    labels_dict = {
        "categories": categories,
        "num_categories": len(categories),
        "bio_tags": bio_tags,
        "tag_to_id": tag_to_id,
        "id_to_tag": {str(k): v for k, v in id_to_tag.items()},
        "num_labels": len(bio_tags)
    }
    
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(categories)} labels to {labels_path}")
    
    logger.info("="*60)
    logger.info("TRAINING DATA PREPARATION COMPLETE")
    logger.info("="*60)


def prepare_test_data(
    config_path: str = "configs/base.yaml",
    output_dir: str = "data/processed"
) -> None:
    """
    Prepare test data:
    - Load raw test dataset
    - Validate
    - Save as JSONL
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for processed data
    """
    config = load_config(config_path)
    setup_logging(log_dir=config["paths"]["logs_dir"])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PREPARING TEST DATA")
    logger.info("="*60)
    
    test_path = Path(config["paths"]["test_data"])
    logger.info(f"Loading raw test data from: {test_path}")
    
    # Load raw data
    test_documents = load_test_dataset(test_path)
    logger.info(f"Loaded {len(test_documents)} test documents")
    
    # Preprocess (validation only, no entity modification)
    logger.info("Preprocessing documents...")
    processed_documents, skipped = preprocess_documents(
        test_documents,
        no_text_mutation=True,
        validate=False
    )
    logger.info(f"Preprocessed: {len(processed_documents)} documents, skipped: {skipped}")
    
    # Save test data
    test_output_path = output_dir / "test.jsonl"
    save_jsonl_documents(processed_documents, test_output_path)
    logger.info(f"Saved {len(processed_documents)} test documents to {test_output_path}")
    
    logger.info("="*60)
    logger.info("TEST DATA PREPARATION COMPLETE")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare PII NER datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--prepare-train",
        action="store_true",
        default=True,
        help="Prepare training data"
    )
    parser.add_argument(
        "--prepare-test",
        action="store_true",
        default=False,
        help="Prepare test data"
    )
    
    args = parser.parse_args()
    
    if args.prepare_train:
        prepare_train_data(args.config, args.output_dir)
    
    if args.prepare_test:
        prepare_test_data(args.config, args.output_dir)


if __name__ == "__main__":
    main()
