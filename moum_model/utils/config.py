"""
Configuration utilities for multi-omics GNN models.

This module provides functions for loading and managing configuration parameters.
"""

import os
import yaml
import json


def load_config(config_path):
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str): Path to configuration file (YAML or JSON)
    
    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found. Using default configuration.")
        return get_default_config()
    
    # Determine file format from extension
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Configuration file must be YAML or JSON format.")
    
    # Merge with default config to ensure all required parameters are present
    default_config = get_default_config()
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    return config


def get_default_config():
    """
    Get default configuration parameters.
    
    Returns:
        dict: Default configuration parameters
    """
    return {
        # Data parameters
        'test_size': 0.2,
        'random_seed': 42,
        'correlation_thresholds': {
            ('gene_expression', 'methylation'): 0.4,
            ('gene_expression', 'copy_number'): 0.4,
            ('methylation', 'copy_number'): 0.4
        },
        
        # Model parameters
        'gnn_type': 'GCN',  # Options: 'GCN', 'GAT', 'SAGE', 'GIN'
        'hidden_dim': 256,
        'embedding_dim': 128,
        'num_gnn_layers': 2,
        'dropout': 0.2,
        'task_type': 'regression',  # Options: 'regression', 'classification'
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 10  # For early stopping
    }


def save_config(config, config_path):
    """
    Save configuration to a file.
    
    Args:
        config (dict): Configuration parameters
        config_path (str): Path to save configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Determine file format from extension
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError("Configuration file must be YAML or JSON format.")
    
    print(f"Configuration saved to {config_path}")
