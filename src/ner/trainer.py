"""Training utilities for NER model."""


import os
from pathlib import Path
from typing import Optional, Dict, Any


import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
)
from transformers import Trainer, TrainingArguments


from src.utils.logging_utils import get_logger
from src.utils.config import load_config, get_labels
from src.ner.labeling import BIOLabeler
from src.ner.model import BertNER
from src.ner.dataset import NERDataset, create_data_collator
from src.data.readers import load_jsonl_documents
from src.metrics.strict_span_f1 import per_category_metrics


logger = get_logger(__name__)



class NERTrainer:
    """Trainer for NER model using HuggingFace Trainer."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "models/ner_checkpoint"
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for checkpoints
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup
        self.device = self.config["training"].get("device", "cuda")
        self.model_name = self.config["ner"]["model_name"]
        
        # Create components
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Get categories and create labeler
        categories = get_labels(config)
        self.labeler = BIOLabeler(categories)
        
        # Create model
        self.model = BertNER(
            self.model_name,
            self.labeler.num_labels
        ).to(self.device)
        
        logger.info(f"Initialized NERTrainer with {self.model_name}")
    
    def train(
        self,
        train_path: str = "data/processed/train.jsonl",
        valid_path: str = "data/processed/valid.jsonl"
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_path: Path to training data
            valid_path: Path to validation data
            
        Returns:
            Training results
        """
        logger.info("Loading datasets...")
        
        # Load datasets
        train_dataset = NERDataset.from_jsonl(
            train_path,
            self.tokenizer,
            self.labeler,
            max_length=self.config["ner"]["tokenizer"]["max_length"]
        )
        
        valid_dataset = NERDataset.from_jsonl(
            valid_path,
            self.tokenizer,
            self.labeler,
            max_length=self.config["ner"]["tokenizer"]["max_length"]
        )
        
        logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"]["epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["eval_batch_size"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            
            # Evaluation and saving
            eval_strategy=self.config["training"]["eval_strategy"],
            eval_steps=self.config["training"]["eval_steps"],
            save_strategy=self.config["training"]["save_strategy"],
            save_steps=self.config["training"]["save_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            load_best_model_at_end=True,
            save_safetensors=False,
            
            # Logging
            logging_steps=self.config["training"]["logging_steps"],
            log_level="info",
            
            # Device
            
            # Reproducibility
            seed=self.config["random_seed"],
        )
        
        # Create trainer
        trainer = Trainer(
          model=self.model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=valid_dataset,
          data_collator=create_data_collator(self.labeler),
          callbacks=[
              EarlyStoppingCallback(
                  early_stopping_patience=self.config["training"]["early_stopping"]["patience"],
                  early_stopping_threshold=0.01
              )
          ]
        )


        
        # Train
        logger.info("Starting training...")
        result = trainer.train()
        
        # Save model and labeler
        trainer.save_model(str(self.output_dir / "model"))
        self.labeler.save(str(self.output_dir / "labeler.json"))
        
        logger.info(f"Training complete. Model saved to {self.output_dir}")
        
        return result



def train_ner_model(
    config_path: str = "configs/base.yaml",
    output_dir: str = "models/ner_checkpoint",
    train_path: str = "data/processed/train.jsonl",
    valid_path: str = "data/processed/valid.jsonl"
) -> None:
    """
    Train NER model from config.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory
        train_path: Path to training data
        valid_path: Path to validation data
    """
    # Load config
    config = load_config(config_path)
    
    # Create and train
    trainer = NERTrainer(config, output_dir)
    trainer.train(train_path, valid_path)
