"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def load_config(config_path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable substitution.
    
    Supports environment variable injection:
        - ${ENV_VAR_NAME} in config will be replaced with os.getenv("ENV_VAR_NAME")
        - Falls back to default value if env var not set
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        # Substitute environment variables
        config = _substitute_env_vars(config)
        
        logger.info(f"Loaded config from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def _substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in config object.
    
    Supports format: ${VAR_NAME} or ${VAR_NAME:default_value}
    
    Args:
        obj: Configuration object (dict, list, or value)
        
    Returns:
        Object with environment variables substituted
    """
    if isinstance(obj, dict):
        return {key: _substitute_env_vars(value) for key, value in obj.items()}
    
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    
    elif isinstance(obj, str):
        # Replace ${VAR_NAME} or ${VAR_NAME:default}
        import re
        pattern = r"\$\{([^:}]+)(?::([^}]*))?\}"
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) or ""
            value = os.getenv(var_name, default_value)
            return value
        
        return re.sub(pattern, replace_var, obj)
    
    else:
        return obj


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Example:
        get_config_value(config, "ner.model_name") -> config['ner']['model_name']
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration dictionary recursively.
    
    Args:
        config: Base configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that required config sections exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises otherwise
        
    Raises:
        ValueError: If required config is missing
    """
    required_keys = ["labels", "paths", "regex", "ner", "training", "inference"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate labels
    if not config["labels"]:
        raise ValueError("No labels defined in config")
    
    return True


def get_labels(config: Dict[str, Any]) -> list[str]:
    """
    Get list of entity labels from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of label names (categories)
    """
    labels = list(config.get("labels", {}).keys())
    if not labels:
        raise ValueError("No labels found in config")
    return labels


def get_label_to_id(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Get mapping from label name to ID.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping label names to integer IDs
    """
    labels = get_labels(config)
    return {label: idx for idx, label in enumerate(labels)}


def get_id_to_label(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Get mapping from label ID to name.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping integer IDs to label names
    """
    label_to_id = get_label_to_id(config)
    return {v: k for k, v in label_to_id.items()}
