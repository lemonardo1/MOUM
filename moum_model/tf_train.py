"""
Training script for GNN-based multi-omics integration models in TensorFlow.

This script provides functionality to train and evaluate GNN-based models
for integrating multiple types of omics data using TensorFlow.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.tf_gnn_integration import MultiOmicsGNN, HeterogeneousOmicsGNN
from data.data_processor import MultiOmicsDataset, load_sample_data
from utils.config import load_config


def train_model(model, train_data, train_response, edge_indices_dict=None, 
                val_data=None, val_response=None, config=None):
    """
    Train the GNN-based multi-omics integration model.
    
    Args:
        model (tf.keras.Model): GNN model to train
        train_data (dict): Dictionary of training data for each omics type
        train_response (tf.Tensor): Training response values
        edge_indices_dict (dict, optional): Dictionary of edge indices for heterogeneous GNN
        val_data (dict, optional): Dictionary of validation data for each omics type
        val_response (tf.Tensor, optional): Validation response values
        config (dict, optional): Configuration parameters. Defaults to None.
    
    Returns:
        tuple: Trained model and training history
    """
    if config is None:
        config = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 10,
            'task_type': 'regression'
        }
    # log config
    print(f"Training config: {config}")
    
    # Define loss function based on task type
    if config['task_type'] == 'classification':
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:  # regression
        loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Optimizer
    optimizer = optimizers.Adam(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss' if val_data is not None else 'loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss' if val_data is not None else 'loss',
        patience=config['patience'],
        restore_best_weights=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_data is not None else None,
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training step
        with tf.GradientTape() as tape:
            if isinstance(model, HeterogeneousOmicsGNN):
                predictions = model(train_data, edge_indices_dict, training=True)
            else:
                predictions = model(train_data, training=True)
            
            loss = loss_fn(train_response, predictions)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Record training loss
        history['train_loss'].append(loss.numpy())
        history['learning_rates'].append(optimizer.learning_rate.numpy())
        
        # Validation
        if val_data is not None and val_response is not None:
            if isinstance(model, HeterogeneousOmicsGNN):
                val_predictions = model(val_data, edge_indices_dict, training=False)
            else:
                val_predictions = model(val_data, training=False)
            
            val_loss = loss_fn(val_response, val_predictions)
            history['val_loss'].append(val_loss.numpy())
            
            # Print progress
            print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                  f"Train Loss: {loss.numpy():.4f}, "
                  f"Val Loss: {val_loss.numpy():.4f}")
            
            # Learning rate scheduling
            lr_scheduler.on_epoch_end(epoch, {'val_loss': val_loss.numpy()})
            
            # Early stopping
            early_stopping.on_epoch_end(epoch, {'val_loss': val_loss.numpy()})
            if early_stopping.stopped_epoch > 0:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # Print progress
            print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                  f"Train Loss: {loss.numpy():.4f}")
    
    return model, history


def evaluate_model(model, test_data, test_response, edge_indices_dict=None, task_type='regression'):
    """
    Evaluate the trained model.
    
    Args:
        model (tf.keras.Model): Trained GNN model
        test_data (dict): Dictionary of test data for each omics type
        test_response (tf.Tensor): Test response values
        edge_indices_dict (dict, optional): Dictionary of edge indices for heterogeneous GNN
        task_type (str, optional): Type of task ('regression' or 'classification')
    
    Returns:
        dict: Evaluation metrics
    """
    # Evaluation mode
    if isinstance(model, HeterogeneousOmicsGNN) and edge_indices_dict is not None:
        predictions = model(test_data, edge_indices_dict, training=False)
    else:
        predictions = model(test_data, training=False)
    
    # Convert to numpy for evaluation
    predictions = predictions.numpy()
    test_response = test_response.numpy()
    
    # Calculate metrics based on task type
    metrics = {}
    if task_type == 'regression':
        metrics['mse'] = mean_squared_error(test_response, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(test_response, predictions)
        
        print(f"Test MSE: {metrics['mse']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test R²: {metrics['r2']:.4f}")
    else:  # classification
        # Ensure binary predictions for classification
        binary_preds = (predictions > 0.5).astype(int)
        
        metrics['accuracy'] = accuracy_score(test_response, binary_preds)
        
        # Calculate ROC AUC if applicable (requires probability scores)
        if test_response.shape[1] == 1:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(test_response, predictions)
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
        else:  # Multi-class
            # Calculate ROC AUC for each class
            metrics['roc_auc'] = []
            for i in range(test_response.shape[1]):
                try:
                    metrics['roc_auc'].append(roc_auc_score(test_response[:, i], predictions[:, i]))
                except:
                    metrics['roc_auc'].append(np.nan)
            
            metrics['roc_auc_mean'] = np.nanmean(metrics['roc_auc'])
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Mean ROC AUC: {metrics['roc_auc_mean']:.4f}")
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss history.
    
    Args:
        history (dict): Training history from train_model function
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if history['val_loss'] is not None:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='Train GNN-based multi-omics integration model')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/omics_data',
                        help='Directory containing omics data files')
    parser.add_argument('--response_file', type=str, default='data/omics_data/drug_response.csv',
                        help='Path to drug response or target variable file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data if true')
    parser.add_argument('--model_type', type=str, default='homogeneous',
                        choices=['homogeneous', 'heterogeneous'],
                        help='Type of GNN model to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate data
    print("Loading data...")
    if args.use_synthetic:
        dataset = load_sample_data(args.data_dir, create_synthetic=True)
    else:
        dataset = MultiOmicsDataset(
            data_dir=args.data_dir,
            response_file=args.response_file,
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_seed', 42)
        )
    
    # Preprocess data
    print("Preprocessing data...")
    dataset.preprocess_data()
    
    # Get train and test data
    train_data, train_response = dataset.get_train_data()
    test_data, test_response = dataset.get_test_data()
    
    # Convert data to TensorFlow tensors
    train_data = {k: tf.convert_to_tensor(v) for k, v in train_data.items()}
    train_response = tf.convert_to_tensor(train_response)
    test_data = {k: tf.convert_to_tensor(v) for k, v in test_data.items()}
    test_response = tf.convert_to_tensor(test_response)
    
    # Create adjacency matrices for heterogeneous GNN
    edge_indices_dict = None
    if args.model_type == 'heterogeneous':
        print("Creating adjacency matrices...")
        edge_indices_dict = dataset.create_adjacency_matrices(
            thresholds=config.get('correlation_thresholds', {})
        )
        edge_indices_dict = {
            k: tf.convert_to_tensor(v) for k, v in edge_indices_dict.items()
        }
    
    # Create model
    print(f"Creating {args.model_type} GNN model...")
    if args.model_type == 'homogeneous':
        model = MultiOmicsGNN(
            omics_dims=dataset.feature_dims,
            hidden_dim=config.get('hidden_dim', 256),
            embedding_dim=config.get('embedding_dim', 128),
            gnn_type=config.get('gnn_type', 'GCN'),
            num_gnn_layers=config.get('num_gnn_layers', 2),
            dropout=config.get('dropout', 0.2),
            task_type=config.get('task_type', 'regression'),
            num_tasks=train_response.shape[1] if len(train_response.shape) > 1 else 1
        )
    else:  # heterogeneous
        model = HeterogeneousOmicsGNN(
            omics_dims=dataset.feature_dims,
            hidden_dim=config.get('hidden_dim', 256),
            embedding_dim=config.get('embedding_dim', 128),
            num_gnn_layers=config.get('num_gnn_layers', 2),
            dropout=config.get('dropout', 0.2),
            task_type=config.get('task_type', 'regression'),
            num_tasks=train_response.shape[1] if len(train_response.shape) > 1 else 1
        )
    
    # Train model
    print("Training model...")
    trained_model, history = train_model(
        model,
        train_data,
        train_response,
        edge_indices_dict=edge_indices_dict,
        config=config
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        trained_model,
        test_data,
        test_response,
        edge_indices_dict=edge_indices_dict,
        task_type=config.get('task_type', 'regression')
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    
    # Plot training history
    plot_training_history(
        history, 
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Save model
    trained_model.save(os.path.join(args.output_dir, 'model'))
    
    print(f"Model and results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 