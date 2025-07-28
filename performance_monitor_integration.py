"""
Integration module for the Performance Monitoring and Alert System

This module provides integration functions to seamlessly incorporate the performance
monitoring system into the existing MVFouls training pipeline.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from performance_monitor import PerformanceMonitor, create_performance_monitor, AlertSeverity
import os
import json


# Class name mappings (should match the existing training pipeline)
ACTION_CLASS_NAMES = {
    0: "Tackling",
    1: "Standing tackling", 
    2: "High leg",
    3: "Holding",
    4: "Pushing",
    5: "Elbowing",
    6: "Challenge",
    7: "Dive"
}

SEVERITY_CLASS_NAMES = {
    0: "No Offence",
    1: "Offence + No Card",
    2: "Offence + Yellow Card", 
    3: "Offence + Red Card"
}


def initialize_performance_monitor(log_dir: str = "performance_logs",
                                 custom_thresholds: Optional[Dict[str, float]] = None) -> PerformanceMonitor:
    """
    Initialize the performance monitor for MVFouls training
    
    Args:
        log_dir: Directory to save performance logs
        custom_thresholds: Custom alert thresholds
        
    Returns:
        Configured PerformanceMonitor instance
    """
    # Default thresholds optimized for MVFouls dataset
    default_thresholds = {
        'zero_recall_epochs': 3,
        'performance_drop_threshold': 0.1,
        'plateau_epochs': 5,
        'min_acceptable_recall': 0.05,
        'target_combined_recall': 0.45,  
        'gradient_norm_threshold': 10.0,
        'loss_divergence_threshold': 2.0
    }
    
    if custom_thresholds:
        default_thresholds.update(custom_thresholds)
    
    monitor = PerformanceMonitor(
        action_class_names=ACTION_CLASS_NAMES,
        severity_class_names=SEVERITY_CLASS_NAMES,
        log_dir=log_dir,
        alert_thresholds=default_thresholds,
        trend_window=5
    )
    
    print(f"Performance monitor initialized with log directory: {log_dir}")
    return monitor


def update_monitor_from_validation(monitor: PerformanceMonitor,
                                 epoch: int,
                                 model: torch.nn.Module,
                                 val_dataloader: torch.utils.data.DataLoader,
                                 criterion_action: torch.nn.Module,
                                 criterion_severity: torch.nn.Module,
                                 device: torch.device,
                                 optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Update performance monitor with validation results
    
    Args:
        monitor: PerformanceMonitor instance
        epoch: Current epoch number
        model: The model being trained
        val_dataloader: Validation data loader
        criterion_action: Action loss function
        criterion_severity: Severity loss function
        device: Device for computation
        optimizer: Optimizer (for learning rate)
        
    Returns:
        Dictionary with performance metrics and corrective actions
    """
    model.eval()
    
    all_action_preds = []
    all_action_targets = []
    all_severity_preds = []
    all_severity_targets = []
    total_loss_action = 0.0
    total_loss_severity = 0.0
    num_batches = 0
    
    # Calculate gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    gradient_norm = total_norm ** (1. / 2)
    
    with torch.no_grad():
        for batch_idx, (videos, action_labels, severity_labels, _) in enumerate(val_dataloader):
            # Move data to device
            videos = [v.to(device) for v in videos]
            action_labels = action_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            # Forward pass
            action_logits, severity_logits = model(videos)
            
            # Calculate losses
            loss_action = criterion_action(action_logits, action_labels)
            loss_severity = criterion_severity(severity_logits, severity_labels)
            
            total_loss_action += loss_action.item()
            total_loss_severity += loss_severity.item()
            num_batches += 1
            
            # Collect predictions and targets
            all_action_preds.append(action_logits.cpu())
            all_action_targets.append(action_labels.cpu())
            all_severity_preds.append(severity_logits.cpu())
            all_severity_targets.append(severity_labels.cpu())
    
    # Concatenate all predictions and targets
    action_predictions = torch.cat(all_action_preds, dim=0)
    action_targets = torch.cat(all_action_targets, dim=0)
    severity_predictions = torch.cat(all_severity_preds, dim=0)
    severity_targets = torch.cat(all_severity_targets, dim=0)
    
    # Calculate average losses
    avg_loss_action = total_loss_action / num_batches
    avg_loss_severity = total_loss_severity / num_batches
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Update monitor
    epoch_metrics = monitor.update_metrics(
        epoch=epoch,
        action_predictions=action_predictions,
        action_targets=action_targets,
        severity_predictions=severity_predictions,
        severity_targets=severity_targets,
        loss_action=avg_loss_action,
        loss_severity=avg_loss_severity,
        learning_rate=current_lr,
        gradient_norm=gradient_norm
    )
    
    # Get recent alerts
    recent_alerts = [alert for alert in monitor.alerts_history 
                    if alert.epoch == epoch]
    
    # Generate corrective actions
    corrective_actions = monitor.trigger_corrective_actions(recent_alerts)
    
    # Create confusion matrices
    action_pred_classes = torch.argmax(action_predictions, dim=1).numpy()
    action_true_classes = torch.argmax(action_targets, dim=1).numpy()
    severity_pred_classes = torch.argmax(severity_predictions, dim=1).numpy()
    severity_true_classes = torch.argmax(severity_targets, dim=1).numpy()
    
    # Save confusion matrices
    action_cm_path = monitor.create_confusion_matrix_plot(
        action_true_classes, action_pred_classes,
        list(ACTION_CLASS_NAMES.values()), 'action', epoch
    )
    
    severity_cm_path = monitor.create_confusion_matrix_plot(
        severity_true_classes, severity_pred_classes,
        list(SEVERITY_CLASS_NAMES.values()), 'severity', epoch
    )
    
    # Create trend plots
    trend_plot_path = monitor.create_performance_trend_plot()
    
    model.train()  # Return to training mode
    
    return {
        'epoch_metrics': epoch_metrics,
        'alerts': recent_alerts,
        'corrective_actions': corrective_actions,
        'confusion_matrices': {
            'action': action_cm_path,
            'severity': severity_cm_path
        },
        'trend_plot': trend_plot_path,
        'gradient_norm': gradient_norm
    }


def apply_corrective_actions(corrective_actions: Dict[str, Any],
                           sampler_weights: Optional[List[float]] = None,
                           loss_weights_action: Optional[torch.Tensor] = None,
                           loss_weights_severity: Optional[torch.Tensor] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Apply corrective actions suggested by the performance monitor
    
    Args:
        corrective_actions: Actions suggested by the monitor
        sampler_weights: Current sampling weights
        loss_weights_action: Current action loss weights
        loss_weights_severity: Current severity loss weights
        optimizer: Optimizer to adjust learning rate
        
    Returns:
        Dictionary with updated parameters
    """
    updates = {
        'sampling_weights_updated': False,
        'loss_weights_updated': False,
        'learning_rate_updated': False,
        'early_stopping_triggered': False,
        'checkpoint_revert_needed': False,
        'synthetic_data_needed': []
    }
    
    # Update sampling weights
    if corrective_actions['sampling_weights'] and sampler_weights is not None:
        for class_id, weight_multiplier in corrective_actions['sampling_weights'].items():
            # Apply weight multiplier to affected classes
            # This would need to be integrated with the actual sampler
            print(f"Suggested: Increase sampling weight for class {class_id} by {weight_multiplier}x")
        updates['sampling_weights_updated'] = True
    
    # Update loss weights
    if corrective_actions['loss_weights']:
        if loss_weights_action is not None:
            for class_id, weight_multiplier in corrective_actions['loss_weights'].items():
                if class_id < len(loss_weights_action):
                    loss_weights_action[class_id] *= weight_multiplier
                    print(f"Updated action loss weight for class {class_id}: {loss_weights_action[class_id]:.3f}")
        
        if loss_weights_severity is not None:
            for class_id, weight_multiplier in corrective_actions['loss_weights'].items():
                if class_id < len(loss_weights_severity):
                    loss_weights_severity[class_id] *= weight_multiplier
                    print(f"Updated severity loss weight for class {class_id}: {loss_weights_severity[class_id]:.3f}")
        
        updates['loss_weights_updated'] = True
    
    # Update learning rate
    if corrective_actions['learning_rate_adjustment'] != 1.0 and optimizer is not None:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= corrective_actions['learning_rate_adjustment']
            print(f"Learning rate adjusted: {old_lr:.6f} -> {param_group['lr']:.6f}")
        updates['learning_rate_updated'] = True
    
    # Handle early stopping
    if corrective_actions['early_stopping']:
        print("Early stopping suggested due to performance plateau")
        updates['early_stopping_triggered'] = True
    
    # Handle checkpoint revert
    if corrective_actions['checkpoint_revert']:
        print("Checkpoint revert suggested due to training instability")
        updates['checkpoint_revert_needed'] = True
    
    # Handle synthetic data generation
    if corrective_actions['synthetic_data_generation']:
        updates['synthetic_data_needed'] = corrective_actions['synthetic_data_generation']
        print(f"Synthetic data generation suggested for classes: {corrective_actions['synthetic_data_generation']}")
    
    return updates


def save_performance_checkpoint(monitor: PerformanceMonitor,
                              epoch: int,
                              model_state_dict: Dict[str, Any],
                              optimizer_state_dict: Dict[str, Any],
                              scheduler_state_dict: Optional[Dict[str, Any]] = None,
                              save_path: str = "performance_checkpoint.pth") -> str:
    """
    Save a checkpoint that includes performance monitoring data
    
    Args:
        monitor: PerformanceMonitor instance
        epoch: Current epoch
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        scheduler_state_dict: Scheduler state dictionary
        save_path: Path to save checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'performance_metrics': [m.to_dict() for m in monitor.epoch_metrics_history],
        'performance_alerts': [a.to_dict() for a in monitor.alerts_history],
        'best_combined_recall': monitor.best_combined_recall,
        'alert_thresholds': monitor.alert_thresholds
    }
    
    if scheduler_state_dict is not None:
        checkpoint['scheduler_state_dict'] = scheduler_state_dict
    
    torch.save(checkpoint, save_path)
    print(f"Performance checkpoint saved to {save_path}")
    
    return save_path


def load_performance_checkpoint(checkpoint_path: str,
                              monitor: PerformanceMonitor,
                              model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              scheduler: Optional[Any] = None) -> int:
    """
    Load a checkpoint with performance monitoring data
    
    Args:
        checkpoint_path: Path to checkpoint file
        monitor: PerformanceMonitor instance to restore
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        
    Returns:
        Epoch number to resume from
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore performance monitoring data
    if 'performance_metrics' in checkpoint:
        # Reconstruct epoch metrics history
        from performance_monitor import EpochMetrics, ClassMetrics
        monitor.epoch_metrics_history = []
        
        for metrics_dict in checkpoint['performance_metrics']:
            # Reconstruct ClassMetrics objects
            action_metrics = {}
            for class_id, class_dict in metrics_dict['action_metrics'].items():
                action_metrics[int(class_id)] = ClassMetrics(**class_dict)
            
            severity_metrics = {}
            for class_id, class_dict in metrics_dict['severity_metrics'].items():
                severity_metrics[int(class_id)] = ClassMetrics(**class_dict)
            
            # Reconstruct EpochMetrics
            epoch_metrics = EpochMetrics(
                epoch=metrics_dict['epoch'],
                timestamp=datetime.fromisoformat(metrics_dict['timestamp']),
                action_metrics=action_metrics,
                severity_metrics=severity_metrics,
                macro_recall_action=metrics_dict['macro_recall_action'],
                macro_recall_severity=metrics_dict['macro_recall_severity'],
                combined_macro_recall=metrics_dict['combined_macro_recall'],
                loss_action=metrics_dict['loss_action'],
                loss_severity=metrics_dict['loss_severity'],
                total_loss=metrics_dict['total_loss'],
                learning_rate=metrics_dict['learning_rate']
            )
            
            monitor.epoch_metrics_history.append(epoch_metrics)
    
    # Restore alerts history
    if 'performance_alerts' in checkpoint:
        from performance_monitor import PerformanceAlert, AlertType, AlertSeverity
        from datetime import datetime
        
        monitor.alerts_history = []
        for alert_dict in checkpoint['performance_alerts']:
            alert = PerformanceAlert(
                alert_type=AlertType(alert_dict['alert_type']),
                severity=AlertSeverity(alert_dict['severity']),
                affected_classes=alert_dict['affected_classes'],
                message=alert_dict['message'],
                suggested_actions=alert_dict['suggested_actions'],
                timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                epoch=alert_dict['epoch'],
                metrics=alert_dict['metrics']
            )
            monitor.alerts_history.append(alert)
    
    # Restore best metrics
    if 'best_combined_recall' in checkpoint:
        monitor.best_combined_recall = checkpoint['best_combined_recall']
    
    print(f"Performance checkpoint loaded from {checkpoint_path}")
    print(f"Restored {len(monitor.epoch_metrics_history)} epochs of metrics")
    print(f"Restored {len(monitor.alerts_history)} alerts")
    
    return checkpoint['epoch'] + 1


def generate_training_summary(monitor: PerformanceMonitor,
                            save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive training summary report
    
    Args:
        monitor: PerformanceMonitor instance
        save_path: Optional path to save the report
        
    Returns:
        Dictionary with comprehensive training summary
    """
    report = monitor.generate_performance_report()
    
    # Add additional analysis
    if monitor.epoch_metrics_history:
        latest_metrics = monitor.epoch_metrics_history[-1]
        
        # Calculate improvement metrics
        if len(monitor.epoch_metrics_history) > 1:
            first_metrics = monitor.epoch_metrics_history[0]
            improvement = {
                'combined_recall_improvement': latest_metrics.combined_macro_recall - first_metrics.combined_macro_recall,
                'action_recall_improvement': latest_metrics.macro_recall_action - first_metrics.macro_recall_action,
                'severity_recall_improvement': latest_metrics.macro_recall_severity - first_metrics.macro_recall_severity
            }
            report['improvement_analysis'] = improvement
        
        # Identify problematic classes
        problematic_classes = {
            'zero_recall_action': [],
            'zero_recall_severity': [],
            'low_recall_action': [],
            'low_recall_severity': []
        }
        
        for class_id, metrics in latest_metrics.action_metrics.items():
            if metrics.recall == 0.0:
                problematic_classes['zero_recall_action'].append(metrics.class_name)
            elif metrics.recall < 0.1:
                problematic_classes['low_recall_action'].append(metrics.class_name)
        
        for class_id, metrics in latest_metrics.severity_metrics.items():
            if metrics.recall == 0.0:
                problematic_classes['zero_recall_severity'].append(metrics.class_name)
            elif metrics.recall < 0.1:
                problematic_classes['low_recall_severity'].append(metrics.class_name)
        
        report['problematic_classes'] = problematic_classes
        
        # Alert summary
        alert_summary = {
            'total_alerts': len(monitor.alerts_history),
            'alerts_by_type': {},
            'alerts_by_severity': {}
        }
        
        for alert in monitor.alerts_history:
            alert_type = alert.alert_type.value
            alert_severity = alert.severity.value
            
            alert_summary['alerts_by_type'][alert_type] = alert_summary['alerts_by_type'].get(alert_type, 0) + 1
            alert_summary['alerts_by_severity'][alert_severity] = alert_summary['alerts_by_severity'].get(alert_severity, 0) + 1
        
        report['alert_summary'] = alert_summary
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Training summary saved to {save_path}")
    
    return report


def check_target_achievement(monitor: PerformanceMonitor) -> Dict[str, Any]:
    """
    Check if training targets have been achieved
    
    Args:
        monitor: PerformanceMonitor instance
        
    Returns:
        Dictionary with target achievement status
    """
    if not monitor.epoch_metrics_history:
        return {'error': 'No metrics available'}
    
    latest_metrics = monitor.epoch_metrics_history[-1]
    target_recall = monitor.alert_thresholds['target_combined_recall']
    
    # Check main target
    target_achieved = latest_metrics.combined_macro_recall >= target_recall
    
    # Check individual class targets from requirements
    minority_targets = {
        'Pushing': 0.15,    # Action class 4
        'Dive': 0.10,       # Action class 7  
        'Red Card': 0.20    # Severity class 3
    }
    
    minority_achievement = {}
    
    # Check Pushing (action class 4)
    if 4 in latest_metrics.action_metrics:
        minority_achievement['Pushing'] = {
            'current': latest_metrics.action_metrics[4].recall,
            'target': minority_targets['Pushing'],
            'achieved': latest_metrics.action_metrics[4].recall >= minority_targets['Pushing']
        }
    
    # Check Dive (action class 7)
    if 7 in latest_metrics.action_metrics:
        minority_achievement['Dive'] = {
            'current': latest_metrics.action_metrics[7].recall,
            'target': minority_targets['Dive'],
            'achieved': latest_metrics.action_metrics[7].recall >= minority_targets['Dive']
        }
    
    # Check Red Card (severity class 3)
    if 3 in latest_metrics.severity_metrics:
        minority_achievement['Red Card'] = {
            'current': latest_metrics.severity_metrics[3].recall,
            'target': minority_targets['Red Card'],
            'achieved': latest_metrics.severity_metrics[3].recall >= minority_targets['Red Card']
        }
    
    return {
        'main_target': {
            'current': latest_metrics.combined_macro_recall,
            'target': target_recall,
            'achieved': target_achieved
        },
        'minority_targets': minority_achievement,
        'all_targets_achieved': target_achieved and all(
            m['achieved'] for m in minority_achievement.values()
        )
    }


# Example integration with existing training loop
def example_training_integration():
    """
    Example of how to integrate the performance monitor into existing training
    """
    print("Example integration code:")
    print("""
    # Initialize performance monitor
    monitor = initialize_performance_monitor("performance_logs")
    
    # In your training loop, after each validation epoch:
    performance_results = update_monitor_from_validation(
        monitor=monitor,
        epoch=epoch,
        model=model,
        val_dataloader=val_dataloader,
        criterion_action=criterion_action,
        criterion_severity=criterion_severity,
        device=device,
        optimizer=optimizer
    )
    
    # Apply corrective actions if needed
    if performance_results['alerts']:
        updates = apply_corrective_actions(
            performance_results['corrective_actions'],
            sampler_weights=sample_weights,
            loss_weights_action=action_class_weights,
            loss_weights_severity=severity_class_weights,
            optimizer=optimizer
        )
        
        # Handle early stopping
        if updates['early_stopping_triggered']:
            print("Early stopping triggered by performance monitor")
            break
    
    # Check target achievement
    target_status = check_target_achievement(monitor)
    if target_status['all_targets_achieved']:
        print("All training targets achieved!")
        break
    
    # Save performance checkpoint
    if epoch % 5 == 0:
        save_performance_checkpoint(
            monitor, epoch, model.state_dict(), 
            optimizer.state_dict(), scheduler.state_dict()
        )
    """)


if __name__ == "__main__":
    example_training_integration()