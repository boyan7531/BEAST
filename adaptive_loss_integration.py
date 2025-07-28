"""
Integration module for Adaptive Loss System with existing training pipeline.
Provides compatibility layer between the new adaptive loss system and current training code.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging

from adaptive_loss_system import (
    DynamicLossSystem, 
    ClassPerformanceMetrics, 
    LossConfig,
    AdaptiveFocalLoss,
    ClassBalanceLoss
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class name mappings for better logging
ACTION_CLASS_NAMES = [
    "Tackling", "Standing tackling", "High leg", "Holding", 
    "Pushing", "Elbowing", "Challenge", "Dive"
]

SEVERITY_CLASS_NAMES = [
    "No offence", "Offence + No card", "Offence + Yellow card", "Offence + Red card"
]


class AdaptiveLossWrapper:
    """
    Wrapper class that provides compatibility with existing training code.
    Allows gradual migration from current loss functions to adaptive loss system.
    """
    
    def __init__(self,
                 num_action_classes: int = 8,
                 num_severity_classes: int = 4,
                 use_adaptive_system: bool = True,
                 config: LossConfig = None):
        """
        Initialize adaptive loss wrapper.
        
        Args:
            num_action_classes: Number of action classes
            num_severity_classes: Number of severity classes
            use_adaptive_system: Whether to use the new adaptive system
            config: Loss configuration
        """
        self.num_action_classes = num_action_classes
        self.num_severity_classes = num_severity_classes
        self.use_adaptive_system = use_adaptive_system
        
        if use_adaptive_system:
            # Initialize the new adaptive loss system
            self.loss_system = DynamicLossSystem(
                num_action_classes=num_action_classes,
                num_severity_classes=num_severity_classes,
                config=config,
                action_class_names=ACTION_CLASS_NAMES,
                severity_class_names=SEVERITY_CLASS_NAMES
            )
            logger.info("Initialized adaptive loss system")
        else:
            # Fallback to individual adaptive focal losses for compatibility
            self.action_focal_loss = AdaptiveFocalLoss(
                num_action_classes,
                config=config,
                class_names=ACTION_CLASS_NAMES
            )
            self.severity_focal_loss = AdaptiveFocalLoss(
                num_severity_classes,
                config=config,
                class_names=SEVERITY_CLASS_NAMES
            )
            logger.info("Initialized individual adaptive focal losses")
        
        # Performance tracking
        self.epoch_count = 0
        self.performance_history = defaultdict(list)
        
    def compute_loss(self,
                    action_logits: torch.Tensor,
                    severity_logits: torch.Tensor,
                    action_targets: torch.Tensor,
                    severity_targets: torch.Tensor,
                    attention_info: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive loss with compatibility for existing training code.
        
        Args:
            action_logits: Action prediction logits
            severity_logits: Severity prediction logits
            action_targets: Action target labels (class indices)
            severity_targets: Severity target labels (class indices)
            attention_info: Optional attention information
            
        Returns:
            Total loss and loss components dictionary
        """
        if self.use_adaptive_system:
            return self.loss_system(
                action_logits, severity_logits,
                action_targets, severity_targets,
                attention_info
            )
        else:
            # Fallback to individual losses
            action_loss = self.action_focal_loss(action_logits, action_targets)
            severity_loss = self.severity_focal_loss(severity_logits, severity_targets)
            
            total_loss = action_loss + severity_loss
            loss_components = {
                'action_focal': action_loss,
                'severity_focal': severity_loss,
                'total': total_loss
            }
            
            return total_loss, loss_components
    
    def update_from_epoch_metrics(self,
                                 action_predictions: torch.Tensor,
                                 severity_predictions: torch.Tensor,
                                 action_targets: torch.Tensor,
                                 severity_targets: torch.Tensor):
        """
        Update loss parameters based on epoch performance metrics.
        
        Args:
            action_predictions: Action predictions from the epoch
            severity_predictions: Severity predictions from the epoch
            action_targets: Action ground truth labels
            severity_targets: Severity ground truth labels
        """
        self.epoch_count += 1
        
        # Calculate performance metrics
        action_metrics = self._calculate_class_metrics(
            action_predictions, action_targets, self.num_action_classes, 'action'
        )
        severity_metrics = self._calculate_class_metrics(
            severity_predictions, severity_targets, self.num_severity_classes, 'severity'
        )
        
        # Update loss parameters
        if self.use_adaptive_system:
            self.loss_system.update_from_metrics(action_metrics, severity_metrics)
        else:
            self.action_focal_loss.update_parameters(action_metrics)
            self.severity_focal_loss.update_parameters(severity_metrics)
        
        # Log performance improvements
        self._log_performance_update(action_metrics, severity_metrics)
    
    def _calculate_class_metrics(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                num_classes: int,
                                task_type: str) -> Dict[int, ClassPerformanceMetrics]:
        """Calculate per-class performance metrics"""
        # Convert to numpy for sklearn metrics
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        else:
            pred_classes = predictions.cpu().numpy()
        
        target_classes = targets.cpu().numpy()
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            target_classes, pred_classes, 
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )
        
        # Create metrics dictionary
        metrics = {}
        for class_id in range(num_classes):
            # Update performance history
            history_key = f"{task_type}_class_{class_id}_recall"
            self.performance_history[history_key].append(recall[class_id])
            
            # Calculate epochs since improvement
            recall_history = self.performance_history[history_key]
            epochs_since_improvement = 0
            if len(recall_history) > 1:
                best_recall = max(recall_history[:-1])
                if recall[class_id] <= best_recall:
                    # Find how many epochs since last improvement
                    for i in range(len(recall_history) - 1, 0, -1):
                        if recall_history[i] <= recall_history[i-1]:
                            epochs_since_improvement += 1
                        else:
                            break
            
            metrics[class_id] = ClassPerformanceMetrics(
                class_id=class_id,
                recall=recall[class_id],
                precision=precision[class_id],
                f1_score=f1[class_id],
                support=int(support[class_id]),
                epochs_since_improvement=epochs_since_improvement,
                best_recall=max(recall_history) if recall_history else recall[class_id],
                recall_history=recall_history[-10:]  # Keep last 10 epochs
            )
        
        return metrics
    
    def _log_performance_update(self,
                               action_metrics: Dict[int, ClassPerformanceMetrics],
                               severity_metrics: Dict[int, ClassPerformanceMetrics]):
        """Log performance updates and improvements"""
        logger.info(f"Epoch {self.epoch_count} Performance Update:")
        
        # Log action class performance
        logger.info("Action Classes:")
        for class_id, metrics in action_metrics.items():
            class_name = ACTION_CLASS_NAMES[class_id] if class_id < len(ACTION_CLASS_NAMES) else f"class_{class_id}"
            improvement = "↑" if metrics.recall > metrics.best_recall else "→" if metrics.recall == metrics.best_recall else "↓"
            logger.info(f"  {class_name}: recall={metrics.recall:.3f} {improvement} (best={metrics.best_recall:.3f})")
        
        # Log severity class performance
        logger.info("Severity Classes:")
        for class_id, metrics in severity_metrics.items():
            class_name = SEVERITY_CLASS_NAMES[class_id] if class_id < len(SEVERITY_CLASS_NAMES) else f"class_{class_id}"
            improvement = "↑" if metrics.recall > metrics.best_recall else "→" if metrics.recall == metrics.best_recall else "↓"
            logger.info(f"  {class_name}: recall={metrics.recall:.3f} {improvement} (best={metrics.best_recall:.3f})")
        
        # Calculate and log combined macro recall
        action_macro_recall = np.mean([m.recall for m in action_metrics.values()])
        severity_macro_recall = np.mean([m.recall for m in severity_metrics.values()])
        combined_macro_recall = (action_macro_recall + severity_macro_recall) / 2
        
        logger.info(f"Combined Macro Recall: {combined_macro_recall:.3f} (target: 0.45)")
        
        # Log classes needing attention (recall < 10%)
        poor_classes = []
        for class_id, metrics in action_metrics.items():
            if metrics.recall < 0.1:
                class_name = ACTION_CLASS_NAMES[class_id] if class_id < len(ACTION_CLASS_NAMES) else f"action_{class_id}"
                poor_classes.append(f"{class_name}({metrics.recall:.3f})")
        
        for class_id, metrics in severity_metrics.items():
            if metrics.recall < 0.1:
                class_name = SEVERITY_CLASS_NAMES[class_id] if class_id < len(SEVERITY_CLASS_NAMES) else f"severity_{class_id}"
                poor_classes.append(f"{class_name}({metrics.recall:.3f})")
        
        if poor_classes:
            logger.warning(f"Classes needing attention (recall < 10%): {', '.join(poor_classes)}")
    
    def get_criterion_functions(self) -> Tuple[nn.Module, nn.Module]:
        """
        Get criterion functions compatible with existing training code.
        Returns individual loss functions that can be used as drop-in replacements.
        """
        if self.use_adaptive_system:
            # Return wrapper functions that use the adaptive system
            action_criterion = AdaptiveCriterionWrapper(self, 'action')
            severity_criterion = AdaptiveCriterionWrapper(self, 'severity')
        else:
            # Return the individual adaptive focal losses
            action_criterion = self.action_focal_loss
            severity_criterion = self.severity_focal_loss
        
        return action_criterion, severity_criterion
    
    def get_current_config(self) -> Dict:
        """Get current loss configuration for logging/debugging"""
        if self.use_adaptive_system:
            return self.loss_system.get_current_config()
        else:
            return {
                'action_focal_params': self.action_focal_loss.get_current_parameters(),
                'severity_focal_params': self.severity_focal_loss.get_current_parameters()
            }


class AdaptiveCriterionWrapper(nn.Module):
    """
    Wrapper that makes the adaptive loss system compatible with existing training code
    that expects separate criterion functions for action and severity.
    """
    
    def __init__(self, adaptive_wrapper: AdaptiveLossWrapper, task_type: str):
        """
        Initialize criterion wrapper.
        
        Args:
            adaptive_wrapper: The main adaptive loss wrapper
            task_type: 'action' or 'severity'
        """
        super().__init__()
        self.adaptive_wrapper = adaptive_wrapper
        self.task_type = task_type
        
        # Store the other task's last inputs for combined loss computation
        self.last_other_logits = None
        self.last_other_targets = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that integrates with the adaptive loss system.
        
        Args:
            logits: Prediction logits for this task
            targets: Target labels for this task
            
        Returns:
            Loss tensor for this task
        """
        if not self.adaptive_wrapper.use_adaptive_system:
            # Fallback to individual loss
            if self.task_type == 'action':
                return self.adaptive_wrapper.action_focal_loss(logits, targets)
            else:
                return self.adaptive_wrapper.severity_focal_loss(logits, targets)
        
        # For the adaptive system, we need both tasks' inputs
        # This is a simplified approach - in practice, you'd modify the training loop
        # to call the adaptive wrapper directly with both tasks' data
        
        if self.task_type == 'action':
            # Store action data and return a placeholder loss
            # The actual loss will be computed when severity is called
            self.adaptive_wrapper._temp_action_logits = logits
            self.adaptive_wrapper._temp_action_targets = targets
            
            # Return individual action focal loss as fallback
            return self.adaptive_wrapper.loss_system.action_focal_loss(logits, targets)
        
        else:  # severity
            # Compute combined loss if we have both tasks' data
            if hasattr(self.adaptive_wrapper, '_temp_action_logits'):
                action_logits = self.adaptive_wrapper._temp_action_logits
                action_targets = self.adaptive_wrapper._temp_action_targets
                
                total_loss, _ = self.adaptive_wrapper.loss_system(
                    action_logits, logits, action_targets, targets
                )
                
                # Clean up temporary storage
                delattr(self.adaptive_wrapper, '_temp_action_logits')
                delattr(self.adaptive_wrapper, '_temp_action_targets')
                
                # Return half the total loss (since both tasks will add their parts)
                return total_loss * 0.5
            else:
                # Fallback to individual severity loss
                return self.adaptive_wrapper.loss_system.severity_focal_loss(logits, targets)


def create_adaptive_loss_wrapper(use_adaptive: bool = True,
                                config: LossConfig = None) -> AdaptiveLossWrapper:
    """
    Factory function to create adaptive loss wrapper with default configuration.
    
    Args:
        use_adaptive: Whether to use the full adaptive system
        config: Optional loss configuration
        
    Returns:
        Configured AdaptiveLossWrapper instance
    """
    if config is None:
        config = LossConfig(
            initial_gamma=2.0,
            initial_alpha=1.0,
            recall_threshold=0.1,  # 10% recall threshold as per requirements
            weight_increase_factor=1.5,  # 50% increase as per requirements
            adaptation_rate=0.1
        )
    
    wrapper = AdaptiveLossWrapper(
        num_action_classes=8,
        num_severity_classes=4,
        use_adaptive_system=use_adaptive,
        config=config
    )
    
    logger.info(f"Created adaptive loss wrapper (adaptive_system={use_adaptive})")
    return wrapper


def integrate_with_enhanced_model(model, adaptive_wrapper: AdaptiveLossWrapper):
    """
    Integration helper for enhanced model that provides attention information.
    
    Args:
        model: Enhanced MVFouls model
        adaptive_wrapper: Adaptive loss wrapper
        
    Returns:
        Modified forward function that includes loss computation
    """
    original_forward = model.forward
    
    def enhanced_forward_with_loss(x_list, targets_action=None, targets_severity=None):
        """Enhanced forward pass that computes adaptive loss"""
        # Get model outputs
        action_logits, severity_logits, confidence_scores, attention_info = original_forward(
            x_list, return_attention=True
        )
        
        # Compute loss if targets are provided
        if targets_action is not None and targets_severity is not None:
            total_loss, loss_components = adaptive_wrapper.compute_loss(
                action_logits, severity_logits,
                targets_action, targets_severity,
                attention_info
            )
            
            return action_logits, severity_logits, confidence_scores, attention_info, total_loss, loss_components
        else:
            return action_logits, severity_logits, confidence_scores, attention_info
    
    return enhanced_forward_with_loss


if __name__ == "__main__":
    print("Testing Adaptive Loss Integration...")
    
    # Test adaptive loss wrapper
    wrapper = create_adaptive_loss_wrapper(use_adaptive=True)
    print("✓ Adaptive loss wrapper created")
    
    # Test criterion functions
    action_criterion, severity_criterion = wrapper.get_criterion_functions()
    print("✓ Criterion functions obtained")
    
    # Test with dummy data
    batch_size = 4
    action_logits = torch.randn(batch_size, 8, requires_grad=True)
    severity_logits = torch.randn(batch_size, 4, requires_grad=True)
    action_targets = torch.randint(0, 8, (batch_size,))
    severity_targets = torch.randint(0, 4, (batch_size,))
    
    # Test combined loss computation
    try:
        total_loss, loss_components = wrapper.compute_loss(
            action_logits, severity_logits, action_targets, severity_targets
        )
        print(f"✓ Combined loss computation successful: {total_loss.item():.4f}")
        
        # Test backward pass
        total_loss.backward()
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        raise
    
    # Test performance update
    try:
        # Create dummy predictions
        action_preds = torch.softmax(action_logits.detach(), dim=1)
        severity_preds = torch.softmax(severity_logits.detach(), dim=1)
        
        wrapper.update_from_epoch_metrics(
            action_preds, severity_preds, action_targets, severity_targets
        )
        print("✓ Performance metrics update successful")
        
    except Exception as e:
        print(f"✗ Performance update failed: {e}")
        raise
    
    print("\n✓ All integration tests passed!")