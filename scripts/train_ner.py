"""Script to train NER model."""

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.ner.trainer import NERTrainer

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default="data/processed/valid.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ner_checkpoint",
        help="Output directory for model"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    setup_logging(log_dir=config["paths"]["logs_dir"])
    
    logger.info("="*60)
    logger.info("NER MODEL TRAINING")
    logger.info("="*60)
    
    # Verify data exists
    train_path = Path(args.train_data)
    valid_path = Path(args.valid_data)
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return
    
    if not valid_path.exists():
        logger.error(f"Validation data not found: {valid_path}")
        return
    
    # Train
    trainer = NERTrainer(config, args.output_dir)
    result = trainer.train(str(train_path), str(valid_path))
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
