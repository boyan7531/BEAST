"""
Integration utilities for the Smart Rebalancing System

This module provides utilities to integrate the SmartRebalancer with the existing
training pipeline, including metric extraction and parameter updates.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from smart_rebalancer import SmartRebalancer, RebalancingConfig
from collections import Counter

def extract_performance_metrics(all_labels: List[int], 
                               all_predictions: List[int],
                               loss: float,
                               task_type: str,
                               class_names: List[str] = None) -> Dict[str, Any]:
    """
    Extract performance metrics from predictions and labels
    
    Args:
        all_labels: List of true class labels
        all_predictions: List of predicted class labels  
        loss: Current loss value
        task_type: 'action' or 'severity'
        class_names: Optional list of class names for logging
        
    Returns:
        Dictionary containing all performance metrics
    """
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Convert to dictionaries
    per_class_recall_dict = {i: float(recall) for i, recall in enumerate(per_class_recall)}
    per_class_precision_dict = {i: float(precision) for i, precision in enumerate(per_class_precision)}
    per_class_f1_dict = {i: float(f1) for i, f1 in enumerate(per_class_f1)}
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_recall': float(recall),
        'macro_precision': float(precision),
        'macro_f1': float(f1),
        'per_class_recall': per_class_recall_dict,
        'per_class_precision': per_class_precision_dict,
        'per_class_f1': per_class_f1_dict,
        'loss': float(loss)
    }
    
    return metrics

def create_weighted_sampler_from_rebalancer(rebalancer: SmartRebalancer,
                                          severity_labels: List[torch.Tensor],
                                          dataset_length: int) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a WeightedRandomSampler based on rebalancer recommendations
    
    Args:
        rebalancer: SmartRebalancer instance
        severity_labels: List of severity label tensors
        dataset_length: Length of the dataset
        
    Returns:
        WeightedRandomSampler instance
    """
    
    # Get sampling strategy parameters
    sampling_params = rebalancer.get_sampling_strategy_params()
    
    if 'multipliers' not in sampling_params or not sampling_params['multipliers']:
        # Fallback to basic severity-based sampling
        all_severity_labels = torch.cat(severity_labels, dim=0).argmax(dim=1).cpu().numpy()
        severity_counts = Counter(all_severity_labels)
        total_samples = len(all_severity_labels)
        
        sample_weights = []
        for i in range(dataset_length):
            severity_class = all_severity_labels[i]
            freq = severity_counts[severity_class] / total_samples
            
            # Check if this is an excluded class (shouldn't happen in filtered dataset, but safety check)
            if hasattr(rebalancer, 'excluded_severity_classes') and severity_class in rebalancer.excluded_severity_classes:
                weight = 0.01  # Very low weight for excluded classes
            elif freq > 0.5:  # Dominant class
                weight = max(0.5, 1.0 / (freq ** 0.4))
            elif freq > 0.25:  # Major class
                weight = min(1.5, 1.0 / (freq ** 0.6))
            elif freq > 0.05:  # Medium class
                weight = min(3.0, 1.0 / (freq ** 0.8))
            else:  # Minority class
                weight = min(8.0, max(4.0, 1.0 / (freq ** 0.95)))
            
            sample_weights.append(weight)
    else:
        # Use rebalancer's recommended multipliers
        multipliers = sampling_params['multipliers']
        all_severity_labels = torch.cat(severity_labels, dim=0).argmax(dim=1).cpu().numpy()
        
        sample_weights = []
        for i in range(dataset_length):
            severity_class = all_severity_labels[i]
            weight = multipliers.get(severity_class, 1.0)
            sample_weights.append(weight)
    
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

def update_focal_loss_from_rebalancer(rebalancer: SmartRebalancer,
                                    task_type: str,
                                    device: torch.device,
                                    focal_loss_class) -> Any:
    """
    Create or update focal loss based on rebalancer recommendations
    
    Args:
        rebalancer: SmartRebalancer instance
        task_type: 'action' or 'severity'
        device: PyTorch device
        focal_loss_class: FocalLoss class to instantiate
        
    Returns:
        Configured FocalLoss instance
    """
    
    # Get focal loss parameters from rebalancer
    focal_params = rebalancer.get_focal_loss_params(task_type)
    class_weights = rebalancer.get_class_weights(task_type, device)
    
    return focal_loss_class(
        gamma=focal_params['gamma'],
        alpha=focal_params.get('alpha'),
        weight=class_weights,
        label_smoothing=focal_params['label_smoothing']
    )

def get_mixup_params_from_rebalancer(rebalancer: SmartRebalancer,
                                   epoch: int) -> Tuple[bool, float]:
    """
    Get mixup parameters from rebalancer
    
    Args:
        rebalancer: SmartRebalancer instance
        epoch: Current epoch number
        
    Returns:
        Tuple of (should_use_mixup, mixup_alpha)
    """
    
    return rebalancer.should_use_mixup(epoch)

def log_rebalancing_status(rebalancer: SmartRebalancer,
                          epoch: int,
                          logger=None):
    """
    Log current rebalancing status and recommendations
    
    Args:
        rebalancer: SmartRebalancer instance
        epoch: Current epoch number
        logger: Optional logger instance
    """
    
    recommendations = rebalancer.get_strategy_recommendations()
    
    log_msg = f"\n=== Rebalancing Status - Epoch {epoch} ===\n"
    
    # Show excluded classes
    if hasattr(rebalancer, 'excluded_action_classes') and rebalancer.excluded_action_classes:
        log_msg += f"Excluded action classes: {sorted(rebalancer.excluded_action_classes)}\n"
    if hasattr(rebalancer, 'excluded_severity_classes') and rebalancer.excluded_severity_classes:
        log_msg += f"Excluded severity classes: {sorted(rebalancer.excluded_severity_classes)}\n"
    
    # Class weights
    log_msg += f"Action class weights: {[f'{w:.3f}' for w in recommendations['class_weights']['action']]}\n"
    log_msg += f"Severity class weights: {[f'{w:.3f}' for w in recommendations['class_weights']['severity']]}\n"
    
    # Focal loss parameters
    action_focal = recommendations['focal_loss_params']['action']
    severity_focal = recommendations['focal_loss_params']['severity']
    log_msg += f"Action focal loss - gamma: {action_focal['gamma']:.2f}, smoothing: {action_focal['label_smoothing']:.3f}\n"
    log_msg += f"Severity focal loss - gamma: {severity_focal['gamma']:.2f}, smoothing: {severity_focal['label_smoothing']:.3f}\n"
    
    # Mixup
    mixup_info = recommendations['mixup']
    log_msg += f"Mixup: {'Enabled' if mixup_info['use_mixup'] else 'Disabled'}"
    if mixup_info['use_mixup']:
        log_msg += f" (alpha: {mixup_info['alpha']:.2f})"
    log_msg += "\n"
    
    # Sampling strategy
    sampling_info = recommendations['sampling_strategy']
    if 'multipliers' in sampling_info and sampling_info['multipliers']:
        log_msg += f"Sampling multipliers: {sampling_info['multipliers']}\n"
    
    log_msg += "=" * 50
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

def create_rebalancer_from_dataset(train_dataset,
                                 config: RebalancingConfig = None,
                                 save_dir: str = "rebalancing_logs",
                                 excluded_action_classes: set = None,
                                 excluded_severity_classes: set = None) -> SmartRebalancer:
    """
    Create and initialize a SmartRebalancer from a training dataset
    
    Args:
        train_dataset: Training dataset with labels_action_list and labels_severity_list
        config: Optional rebalancing configuration
        save_dir: Directory to save rebalancing logs
        
    Returns:
        Initialized SmartRebalancer instance
    """
    
    rebalancer = SmartRebalancer(
        num_action_classes=8,
        num_severity_classes=4,
        config=config,
        save_dir=save_dir,
        excluded_action_classes=excluded_action_classes,
        excluded_severity_classes=excluded_severity_classes
    )
    
    # Initialize with dataset distributions
    rebalancer.initialize_class_distributions(
        train_dataset.labels_action_list,
        train_dataset.labels_severity_list
    )
    
    return rebalancer

def apply_rebalancer_to_training_step(rebalancer: SmartRebalancer,
                                    epoch: int,
                                    action_criterion,
                                    severity_criterion,
                                    device: torch.device,
                                    focal_loss_class) -> Tuple[Any, Any]:
    """
    Apply rebalancer recommendations to update loss functions for current training step
    
    Args:
        rebalancer: SmartRebalancer instance
        epoch: Current epoch
        action_criterion: Current action loss function
        severity_criterion: Current severity loss function
        device: PyTorch device
        focal_loss_class: FocalLoss class for creating new instances
        
    Returns:
        Tuple of (updated_action_criterion, updated_severity_criterion)
    """
    
    # Get recommendations
    recommendations = rebalancer.get_strategy_recommendations()
    
    if recommendations['use_focal_loss']:
        # Update focal loss parameters
        updated_action_criterion = update_focal_loss_from_rebalancer(
            rebalancer, 'action', device, focal_loss_class
        )
        updated_severity_criterion = update_focal_loss_from_rebalancer(
            rebalancer, 'severity', device, focal_loss_class
        )
        
        return updated_action_criterion, updated_severity_criterion
    else:
        # Use CrossEntropyLoss with updated weights
        action_weights = rebalancer.get_class_weights('action', device)
        severity_weights = rebalancer.get_class_weights('severity', device)
        
        updated_action_criterion = torch.nn.CrossEntropyLoss(weight=action_weights)
        updated_severity_criterion = torch.nn.CrossEntropyLoss(weight=severity_weights)
        
        return updated_action_criterion, updated_severity_criterion

# Class name mappings for logging
ACTION_CLASS_NAMES = [
    "Tackling", "Standing tackling", "High leg", "Holding", 
    "Pushing", "Elbowing", "Challenge", "Dive"
]

SEVERITY_CLASS_NAMES = [
    "No Offence", "Offence + No Card", "Offence + Yellow Card", "Offence + Red Card"
]