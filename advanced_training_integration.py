"""
Integration module for Advanced Training Strategies with MVFouls training system.
Provides seamless integration with existing training code.

This module integrates:
- CosineAnnealingWarmRestarts scheduler with adaptive restarts
- Enhanced gradient accumulation
- Early stopping based on combined macro recall
- Minority class performance monitoring

Requirements: 7.1, 7.2, 7.3, 2.1, 2.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
import json
from datetime import datetime

from advanced_training_strategies import (
    AdvancedTrainingStrategiesManager,
    create_advanced_training_setup
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_performance_metrics_for_strategies(predictions_action: torch.Tensor,
                                             predictions_severity: torch.Tensor,
                                             targets_action: torch.Tensor,
                                             targets_severity: torch.Tensor) -> Dict:
    """
    Extract performance metrics needed for advanced training strategies.
    
    Args:
        predictions_action: Action predictions tensor
        predictions_severity: Severity predictions tensor
        targets_action: Action targets tensor
        targets_severity: Severity targets tensor
        
    Returns:
        Dictionary containing performance metrics
    """
    # Convert predictions to class indices
    pred_action_classes = torch.argmax(predictions_action, dim=1).cpu().numpy()
    pred_severity_classes = torch.argmax(predictions_severity, dim=1).cpu().numpy()
    
    # Convert targets to class indices
    target_action_classes = torch.argmax(targets_action, dim=1).cpu().numpy()
    target_severity_classes = torch.argmax(targets_severity, dim=1).cpu().numpy()
    
    # Calculate per-class metrics for action task
    action_precision, action_recall, action_f1, action_support = precision_recall_fscore_support(
        target_action_classes, pred_action_classes, average=None, zero_division=0
    )
    
    # Calculate per-class metrics for severity task
    severity_precision, severity_recall, severity_f1, severity_support = precision_recall_fscore_support(
        target_severity_classes, pred_severity_classes, average=None, zero_division=0
    )
    
    # Calculate macro averages
    action_macro_recall = np.mean(action_recall)
    severity_macro_recall = np.mean(severity_recall)
    
    # Create metrics dictionary
    metrics = {
        'action_class_recall': action_recall.tolist(),
        'action_class_precision': action_precision.tolist(),
        'action_class_f1': action_f1.tolist(),
        'action_macro_recall': action_macro_recall,
        'action_macro_precision': np.mean(action_precision),
        'action_macro_f1': np.mean(action_f1),
        
        'severity_class_recall': severity_recall.tolist(),
        'severity_class_precision': severity_precision.tolist(),
        'severity_class_f1': severity_f1.tolist(),
        'severity_macro_recall': severity_macro_recall,
        'severity_macro_precision': np.mean(severity_precision),
        'severity_macro_f1': np.mean(severity_f1),
        
        'combined_macro_recall': (action_macro_recall + severity_macro_recall) / 2
    }
    
    return metrics


def create_enhanced_training_loop(model: nn.Module,
                                optimizer: optim.Optimizer,
                                train_dataloader,
                                val_dataloader,
                                criterion_action,
                                criterion_severity,
                                device: torch.device,
                                epochs: int,
                                scaler: torch.cuda.amp.GradScaler,
                                config: Dict = None,
                                save_dir: str = "models") -> Dict:
    """
    Create an enhanced training loop with advanced training strategies.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        criterion_action: Action loss function
        criterion_severity: Severity loss function
        device: Device to run on
        epochs: Number of epochs
        scaler: GradScaler for mixed precision
        config: Configuration for advanced strategies
        save_dir: Directory to save models
        
    Returns:
        Training results dictionary
    """
    logger.info("Setting up enhanced training loop with advanced strategies...")
    
    # Create advanced training strategies manager
    strategies_manager = create_advanced_training_setup(optimizer, config)
    
    # Training history
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'action_macro_recalls': [],
        'severity_macro_recalls': [],
        'combined_macro_recalls': [],
        'learning_rates': [],
        'effective_batch_sizes': [],
        'scheduler_restarts': [],
        'early_stopping_triggered': False,
        'best_epoch': -1,
        'best_combined_recall': -np.inf
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Starting enhanced training loop...")
    logger.info(f"  Total epochs: {epochs}")
    logger.info(f"  Effective batch size: {strategies_manager.get_effective_batch_size(len(next(iter(train_dataloader))[0]))}")
    logger.info(f"  Initial learning rate: {strategies_manager.get_current_lr():.6f}")
    
    for epoch in range(epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Collect predictions for metrics calculation
        all_train_pred_action = []
        all_train_pred_severity = []
        all_train_target_action = []
        all_train_target_severity = []
        
        for batch_idx, (videos, action_labels, severity_labels, action_ids) in enumerate(train_dataloader):
            # Move data to device
            videos = [video.to(device) for video in videos]
            action_labels = action_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                action_logits, severity_logits = model(videos)
                
                # Calculate losses
                loss_action = criterion_action(action_logits, action_labels)
                loss_severity = criterion_severity(severity_logits, severity_labels)
                total_loss = loss_action + loss_severity
                
                # Scale loss for gradient accumulation
                scaled_loss = total_loss / strategies_manager.gradient_accumulator.current_accumulation_steps
            
            # Backward pass
            scaler.scale(scaled_loss).backward()
            
            # Check if should step optimizer or accumulate gradients
            if not strategies_manager.should_accumulate_gradients():
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Mark step completed
                strategies_manager.optimizer_step_completed()
            
            # Accumulate loss and predictions
            train_loss += total_loss.item()
            train_batches += 1
            
            # Store predictions for metrics
            all_train_pred_action.append(action_logits.detach())
            all_train_pred_severity.append(severity_logits.detach())
            all_train_target_action.append(action_labels.detach())
            all_train_target_severity.append(severity_labels.detach())
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                current_lr = strategies_manager.get_current_lr()
                effective_bs = strategies_manager.get_effective_batch_size(len(videos))
                logger.info(f"  Batch {batch_idx + 1}/{len(train_dataloader)}, "
                           f"Loss: {total_loss.item():.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Eff. BS: {effective_bs}")
        
        # Calculate training metrics
        train_pred_action = torch.cat(all_train_pred_action, dim=0)
        train_pred_severity = torch.cat(all_train_pred_severity, dim=0)
        train_target_action = torch.cat(all_train_target_action, dim=0)
        train_target_severity = torch.cat(all_train_target_severity, dim=0)
        
        train_metrics = extract_performance_metrics_for_strategies(
            train_pred_action, train_pred_severity,
            train_target_action, train_target_severity
        )
        
        avg_train_loss = train_loss / train_batches
        logger.info(f"Training - Loss: {avg_train_loss:.4f}, "
                   f"Action Recall: {train_metrics['action_macro_recall']:.4f}, "
                   f"Severity Recall: {train_metrics['severity_macro_recall']:.4f}, "
                   f"Combined: {train_metrics['combined_macro_recall']:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # Collect validation predictions
        all_val_pred_action = []
        all_val_pred_severity = []
        all_val_target_action = []
        all_val_target_severity = []
        
        with torch.no_grad():
            for videos, action_labels, severity_labels, action_ids in val_dataloader:
                # Move data to device
                videos = [video.to(device) for video in videos]
                action_labels = action_labels.to(device)
                severity_labels = severity_labels.to(device)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    action_logits, severity_logits = model(videos)
                    
                    # Calculate losses
                    loss_action = criterion_action(action_logits, action_labels)
                    loss_severity = criterion_severity(severity_logits, severity_labels)
                    total_loss = loss_action + loss_severity
                
                # Accumulate loss and predictions
                val_loss += total_loss.item()
                val_batches += 1
                
                # Store predictions for metrics
                all_val_pred_action.append(action_logits.detach())
                all_val_pred_severity.append(severity_logits.detach())
                all_val_target_action.append(action_labels.detach())
                all_val_target_severity.append(severity_labels.detach())
        
        # Calculate validation metrics
        val_pred_action = torch.cat(all_val_pred_action, dim=0)
        val_pred_severity = torch.cat(all_val_pred_severity, dim=0)
        val_target_action = torch.cat(all_val_target_action, dim=0)
        val_target_severity = torch.cat(all_val_target_severity, dim=0)
        
        val_metrics = extract_performance_metrics_for_strategies(
            val_pred_action, val_pred_severity,
            val_target_action, val_target_severity
        )
        
        avg_val_loss = val_loss / val_batches
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, "
                   f"Action Recall: {val_metrics['action_macro_recall']:.4f}, "
                   f"Severity Recall: {val_metrics['severity_macro_recall']:.4f}, "
                   f"Combined: {val_metrics['combined_macro_recall']:.4f}")
        
        # Step scheduler with validation metrics
        strategies_manager.step_scheduler(epoch, val_metrics)
        
        # Check early stopping
        should_stop = strategies_manager.check_early_stopping(
            model,
            val_metrics['action_macro_recall'],
            val_metrics['severity_macro_recall'],
            val_metrics['action_class_recall'],
            val_metrics['severity_class_recall'],
            epoch
        )
        
        # Update training history
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['action_macro_recalls'].append(val_metrics['action_macro_recall'])
        training_history['severity_macro_recalls'].append(val_metrics['severity_macro_recall'])
        training_history['combined_macro_recalls'].append(val_metrics['combined_macro_recall'])
        training_history['learning_rates'].append(strategies_manager.get_current_lr())
        training_history['effective_batch_sizes'].append(
            strategies_manager.get_effective_batch_size(len(next(iter(train_dataloader))[0]))
        )
        
        # Track scheduler restarts
        if strategies_manager.scheduler.total_restarts > len(training_history['scheduler_restarts']):
            training_history['scheduler_restarts'].append(epoch)
        
        # Update best performance tracking
        if val_metrics['combined_macro_recall'] > training_history['best_combined_recall']:
            training_history['best_combined_recall'] = val_metrics['combined_macro_recall']
            training_history['best_epoch'] = epoch
        
        # Save checkpoint with advanced strategies state
        checkpoint_path = strategies_manager.save_checkpoint(
            epoch, model.state_dict(), optimizer.state_dict()
        )
        
        # Log comprehensive statistics
        comprehensive_stats = strategies_manager.get_comprehensive_stats()
        logger.info(f"Advanced Training Stats:")
        logger.info(f"  Current LR: {comprehensive_stats['scheduler_stats']['current_lr']:.6f}")
        logger.info(f"  Scheduler restarts: {comprehensive_stats['scheduler_stats']['total_restarts']}")
        logger.info(f"  Accumulation steps: {comprehensive_stats['accumulation_stats']['current_accumulation_steps']}")
        logger.info(f"  Early stopping patience: {comprehensive_stats['early_stopping_stats']['patience_counter']}")
        
        # Print minority class performance
        logger.info(f"Minority Class Performance:")
        if len(val_metrics['action_class_recall']) > 7:
            logger.info(f"  Pushing (Action 4): {val_metrics['action_class_recall'][4]:.4f}")
            logger.info(f"  Dive (Action 7): {val_metrics['action_class_recall'][7]:.4f}")
        if len(val_metrics['severity_class_recall']) > 3:
            logger.info(f"  Red Card (Severity 3): {val_metrics['severity_class_recall'][3]:.4f}")
        
        # Check for early stopping
        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            training_history['early_stopping_triggered'] = True
            break
    
    # Save final training history
    history_path = os.path.join(save_dir, "advanced_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Enhanced training loop completed!")
    logger.info(f"  Best combined macro recall: {training_history['best_combined_recall']:.4f} at epoch {training_history['best_epoch'] + 1}")
    logger.info(f"  Total scheduler restarts: {len(training_history['scheduler_restarts'])}")
    logger.info(f"  Early stopping triggered: {training_history['early_stopping_triggered']}")
    
    return training_history


def integrate_with_existing_training(train_script_globals: Dict) -> Dict:
    """
    Integrate advanced training strategies with existing training script.
    
    This function modifies the existing training script's global variables
    to use advanced training strategies.
    
    Args:
        train_script_globals: Global variables from the training script
        
    Returns:
        Configuration dictionary for the integration
    """
    logger.info("Integrating advanced training strategies with existing training script...")
    
    # Extract necessary components from training script
    model = train_script_globals.get('model')
    optimizer = train_script_globals.get('optimizer')
    
    if model is None or optimizer is None:
        raise ValueError("Model and optimizer must be available in training script globals")
    
    # Create advanced training configuration
    config = {
        'scheduler': {
            'T_0': 8,  # Shorter initial restart period for faster adaptation
            'T_mult': 2,
            'eta_min': 1e-7,  # Lower minimum LR for fine-tuning
            'adaptive_restart': True,
            'minority_performance_threshold': 0.03  # Lower threshold for minority classes
        },
        'gradient_accumulation': {
            'base_accumulation_steps': train_script_globals.get('ACCUMULATION_STEPS', 4),
            'max_accumulation_steps': 20,  # Higher max for larger effective batch sizes
            'memory_threshold': 0.80,  # Conservative memory usage
            'adaptive_accumulation': True
        },
        'early_stopping': {
            'patience': 12,  # Reduced patience for faster convergence
            'min_delta': 0.002,  # Smaller delta for fine-grained improvements
            'restore_best_weights': True,
            'minority_class_weight': 3.0  # Higher weight for minority classes
        }
    }
    
    # Create strategies manager
    strategies_manager = create_advanced_training_setup(optimizer, config)
    
    # Store in globals for access by training script
    train_script_globals['strategies_manager'] = strategies_manager
    train_script_globals['advanced_training_config'] = config
    
    logger.info("Advanced training strategies integration complete!")
    logger.info(f"  Effective batch size multiplier: {strategies_manager.gradient_accumulator.current_accumulation_steps}")
    logger.info(f"  Initial learning rate: {strategies_manager.get_current_lr():.6f}")
    logger.info(f"  Early stopping patience: {strategies_manager.early_stopping.patience}")
    
    return config


def replace_scheduler_in_training_script(train_script_globals: Dict):
    """
    Replace the existing scheduler in the training script with advanced scheduler.
    
    Args:
        train_script_globals: Global variables from the training script
    """
    logger.info("Replacing existing scheduler with CosineAnnealingWarmRestarts...")
    
    # Get the strategies manager
    strategies_manager = train_script_globals.get('strategies_manager')
    
    if strategies_manager is None:
        raise ValueError("Advanced training strategies must be integrated first")
    
    # Replace the scheduler in globals
    train_script_globals['scheduler'] = strategies_manager.scheduler
    
    # Add helper functions to globals
    train_script_globals['should_accumulate_gradients'] = strategies_manager.should_accumulate_gradients
    train_script_globals['optimizer_step_completed'] = strategies_manager.optimizer_step_completed
    train_script_globals['check_early_stopping'] = strategies_manager.check_early_stopping
    train_script_globals['get_effective_batch_size'] = strategies_manager.get_effective_batch_size
    train_script_globals['extract_performance_metrics_for_strategies'] = extract_performance_metrics_for_strategies
    
    logger.info("Scheduler replacement complete!")


def create_training_loop_wrapper(original_training_function):
    """
    Create a wrapper for the original training function that adds advanced strategies.
    
    Args:
        original_training_function: Original training function
        
    Returns:
        Wrapped training function with advanced strategies
    """
    def wrapped_training_function(*args, **kwargs):
        logger.info("Starting training with advanced strategies wrapper...")
        
        # Extract necessary arguments
        model = kwargs.get('model') or args[0] if args else None
        optimizer = kwargs.get('optimizer') or args[1] if len(args) > 1 else None
        
        if model is None or optimizer is None:
            logger.warning("Could not extract model and optimizer for advanced strategies")
            return original_training_function(*args, **kwargs)
        
        # Create advanced training setup
        strategies_manager = create_advanced_training_setup(optimizer)
        
        # Add strategies manager to kwargs
        kwargs['strategies_manager'] = strategies_manager
        
        # Call original function with enhanced arguments
        result = original_training_function(*args, **kwargs)
        
        logger.info("Training with advanced strategies completed!")
        
        return result
    
    return wrapped_training_function


# Example usage functions
def demonstrate_integration():
    """Demonstrate how to integrate advanced training strategies."""
    logger.info("Demonstrating advanced training strategies integration...")
    
    # Example of how to modify existing training script
    example_globals = {
        'model': nn.Linear(10, 2),
        'optimizer': optim.AdamW(nn.Linear(10, 2).parameters(), lr=1e-3),
        'ACCUMULATION_STEPS': 4
    }
    
    # Integrate advanced strategies
    config = integrate_with_existing_training(example_globals)
    
    # Replace scheduler
    replace_scheduler_in_training_script(example_globals)
    
    # Now the training script can use:
    # - example_globals['strategies_manager']
    # - example_globals['should_accumulate_gradients']()
    # - example_globals['check_early_stopping'](...)
    # etc.
    
    logger.info("Integration demonstration complete!")
    return config


if __name__ == "__main__":
    # Test the integration
    demonstrate_integration()