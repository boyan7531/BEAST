"""
Advanced Training Strategies Module for MVFouls Performance Optimization.

This module implements sophisticated training strategies including:
- CosineAnnealingWarmRestarts scheduler
- Enhanced gradient accumulation
- Early stopping based on combined macro recall
- Adaptive restart triggers based on minority class performance

Requirements: 7.1, 7.2, 7.3, 2.1, 2.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque
import os
import json
from datetime import datetime
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineAnnealingWarmRestartsScheduler:
    """
    Enhanced CosineAnnealingWarmRestarts scheduler with adaptive restart triggers.
    
    Implements requirement 7.1: Replace current scheduler with CosineAnnealingWarmRestarts
    """
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 T_0: int = 10,
                 T_mult: int = 2,
                 eta_min: float = 1e-6,
                 last_epoch: int = -1,
                 adaptive_restart: bool = True,
                 minority_performance_threshold: float = 0.05):
        """
        Initialize the enhanced scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            T_0: Number of iterations for the first restart
            T_mult: Factor to increase T_i after each restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
            adaptive_restart: Whether to enable adaptive restarts
            minority_performance_threshold: Threshold for minority class performance
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.adaptive_restart = adaptive_restart
        self.minority_performance_threshold = minority_performance_threshold
        
        # Initialize the base scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch
        )
        
        # Tracking variables for adaptive restarts
        self.minority_performance_history = deque(maxlen=5)
        self.epochs_since_restart = 0
        self.total_restarts = 0
        self.last_restart_epoch = -1
        
        logger.info(f"CosineAnnealingWarmRestartsScheduler initialized:")
        logger.info(f"  T_0: {T_0}, T_mult: {T_mult}, eta_min: {eta_min}")
        logger.info(f"  Adaptive restart: {adaptive_restart}")
        logger.info(f"  Minority threshold: {minority_performance_threshold}")
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[Dict] = None):
        """
        Step the scheduler with optional adaptive restart logic.
        
        Args:
            epoch: Current epoch (optional)
            metrics: Performance metrics for adaptive restart decision
        """
        # Check for adaptive restart trigger
        should_restart = False
        
        if self.adaptive_restart and metrics is not None:
            should_restart = self._check_adaptive_restart_trigger(metrics)
        
        if should_restart:
            self._trigger_adaptive_restart(epoch)
        else:
            self.scheduler.step(epoch)
            self.epochs_since_restart += 1
    
    def _check_adaptive_restart_trigger(self, metrics: Dict) -> bool:
        """
        Check if adaptive restart should be triggered based on minority class performance.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if restart should be triggered
        """
        # Extract minority class performance
        minority_recall = self._extract_minority_recall(metrics)
        self.minority_performance_history.append(minority_recall)
        
        # Only consider restart after minimum epochs since last restart
        if self.epochs_since_restart < 5:
            return False
        
        # Check if minority performance has plateaued or degraded
        if len(self.minority_performance_history) >= 3:
            recent_performance = list(self.minority_performance_history)[-3:]
            
            # Check for plateau (no improvement in last 3 epochs)
            is_plateau = all(
                abs(recent_performance[i] - recent_performance[i-1]) < 0.01 
                for i in range(1, len(recent_performance))
            )
            
            # Check for degradation
            is_degrading = (
                recent_performance[-1] < recent_performance[0] - 0.02
            )
            
            # Check if performance is below threshold
            below_threshold = minority_recall < self.minority_performance_threshold
            
            # Trigger restart if conditions are met
            if (is_plateau or is_degrading) and below_threshold:
                logger.info(f"Adaptive restart triggered:")
                logger.info(f"  Minority recall: {minority_recall:.4f}")
                logger.info(f"  Plateau: {is_plateau}, Degrading: {is_degrading}")
                logger.info(f"  Below threshold: {below_threshold}")
                return True
        
        return False
    
    def _extract_minority_recall(self, metrics: Dict) -> float:
        """
        Extract minority class recall from metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Average minority class recall
        """
        minority_recalls = []
        
        # Extract action minority classes (Pushing=4, Dive=7)
        if 'action_class_recall' in metrics:
            action_recalls = metrics['action_class_recall']
            if isinstance(action_recalls, (list, tuple)) and len(action_recalls) > 7:
                minority_recalls.extend([action_recalls[4], action_recalls[7]])
        
        # Extract severity minority class (Red Card=3)
        if 'severity_class_recall' in metrics:
            severity_recalls = metrics['severity_class_recall']
            if isinstance(severity_recalls, (list, tuple)) and len(severity_recalls) > 3:
                minority_recalls.append(severity_recalls[3])
        
        # Return average minority recall
        return np.mean(minority_recalls) if minority_recalls else 0.0
    
    def _trigger_adaptive_restart(self, epoch: Optional[int] = None):
        """
        Trigger an adaptive restart of the scheduler.
        
        Args:
            epoch: Current epoch
        """
        self.total_restarts += 1
        self.last_restart_epoch = epoch if epoch is not None else -1
        self.epochs_since_restart = 0
        
        # Reset the scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=-1
        )
        
        logger.info(f"Adaptive restart #{self.total_restarts} triggered at epoch {epoch}")
        logger.info(f"Learning rate reset to: {self.get_last_lr()[0]:.6f}")
    
    def get_last_lr(self) -> List[float]:
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self) -> Dict:
        """Get scheduler state dictionary."""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'epochs_since_restart': self.epochs_since_restart,
            'total_restarts': self.total_restarts,
            'last_restart_epoch': self.last_restart_epoch,
            'minority_performance_history': list(self.minority_performance_history)
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state dictionary."""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.epochs_since_restart = state_dict.get('epochs_since_restart', 0)
        self.total_restarts = state_dict.get('total_restarts', 0)
        self.last_restart_epoch = state_dict.get('last_restart_epoch', -1)
        
        # Restore performance history
        history = state_dict.get('minority_performance_history', [])
        self.minority_performance_history = deque(history, maxlen=5)


class EnhancedGradientAccumulator:
    """
    Enhanced gradient accumulation with dynamic batch size adjustment.
    
    Implements requirement 7.2: Enhance gradient accumulation for larger effective batch sizes
    """
    
    def __init__(self,
                 base_accumulation_steps: int = 4,
                 max_accumulation_steps: int = 16,
                 memory_threshold: float = 0.85,
                 adaptive_accumulation: bool = True):
        """
        Initialize the gradient accumulator.
        
        Args:
            base_accumulation_steps: Base number of accumulation steps
            max_accumulation_steps: Maximum accumulation steps
            memory_threshold: GPU memory threshold for adaptive adjustment
            adaptive_accumulation: Whether to enable adaptive accumulation
        """
        self.base_accumulation_steps = base_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        self.memory_threshold = memory_threshold
        self.adaptive_accumulation = adaptive_accumulation
        
        self.current_accumulation_steps = base_accumulation_steps
        self.accumulated_steps = 0
        self.total_accumulated_batches = 0
        
        logger.info(f"EnhancedGradientAccumulator initialized:")
        logger.info(f"  Base steps: {base_accumulation_steps}")
        logger.info(f"  Max steps: {max_accumulation_steps}")
        logger.info(f"  Adaptive: {adaptive_accumulation}")
    
    def should_accumulate(self) -> bool:
        """
        Check if gradients should be accumulated (not stepped).
        
        Returns:
            True if should accumulate, False if should step
        """
        self.accumulated_steps += 1
        return self.accumulated_steps < self.current_accumulation_steps
    
    def step_completed(self):
        """Mark that an optimizer step has been completed."""
        self.total_accumulated_batches += self.accumulated_steps
        self.accumulated_steps = 0
        
        # Adjust accumulation steps if adaptive mode is enabled
        if self.adaptive_accumulation:
            self._adjust_accumulation_steps()
    
    def _adjust_accumulation_steps(self):
        """
        Adjust accumulation steps based on GPU memory usage.
        """
        if not torch.cuda.is_available():
            return
        
        try:
            # Get GPU memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_usage_ratio = memory_allocated / memory_reserved if memory_reserved > 0 else 0
            
            # Adjust accumulation steps based on memory usage
            if memory_usage_ratio > self.memory_threshold:
                # High memory usage - reduce accumulation steps
                new_steps = max(2, self.current_accumulation_steps - 1)
            elif memory_usage_ratio < 0.6:
                # Low memory usage - increase accumulation steps
                new_steps = min(self.max_accumulation_steps, self.current_accumulation_steps + 1)
            else:
                # Optimal memory usage - keep current steps
                new_steps = self.current_accumulation_steps
            
            if new_steps != self.current_accumulation_steps:
                logger.info(f"Adjusting accumulation steps: {self.current_accumulation_steps} -> {new_steps}")
                logger.info(f"  Memory usage: {memory_usage_ratio:.2%}")
                self.current_accumulation_steps = new_steps
                
        except Exception as e:
            logger.warning(f"Failed to adjust accumulation steps: {e}")
    
    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """
        Get the effective batch size considering accumulation.
        
        Args:
            base_batch_size: Base batch size
            
        Returns:
            Effective batch size
        """
        return base_batch_size * self.current_accumulation_steps
    
    def get_stats(self) -> Dict:
        """Get accumulation statistics."""
        return {
            'current_accumulation_steps': self.current_accumulation_steps,
            'accumulated_steps': self.accumulated_steps,
            'total_accumulated_batches': self.total_accumulated_batches,
            'adaptive_accumulation': self.adaptive_accumulation
        }


class CombinedMacroRecallEarlyStopping:
    """
    Early stopping based on combined macro recall instead of loss.
    
    Implements requirement 7.3: Early stopping based on combined macro recall
    """
    
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.001,
                 restore_best_weights: bool = True,
                 minority_class_weight: float = 2.0):
        """
        Initialize early stopping based on combined macro recall.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on stop
            minority_class_weight: Weight for minority classes in combined score
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.minority_class_weight = minority_class_weight
        
        self.best_combined_recall = -np.inf
        self.patience_counter = 0
        self.best_model_state = None
        self.should_stop = False
        self.improvement_history = []
        
        logger.info(f"CombinedMacroRecallEarlyStopping initialized:")
        logger.info(f"  Patience: {patience}, Min delta: {min_delta}")
        logger.info(f"  Minority weight: {minority_class_weight}")
    
    def __call__(self, 
                 model: nn.Module,
                 action_macro_recall: float,
                 severity_macro_recall: float,
                 action_class_recalls: List[float],
                 severity_class_recalls: List[float],
                 epoch: int) -> bool:
        """
        Check if training should stop based on combined macro recall.
        
        Args:
            model: PyTorch model
            action_macro_recall: Action task macro recall
            severity_macro_recall: Severity task macro recall
            action_class_recalls: Per-class recalls for action task
            severity_class_recalls: Per-class recalls for severity task
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        # Calculate combined macro recall with minority class weighting
        combined_recall = self._calculate_weighted_combined_recall(
            action_macro_recall, severity_macro_recall,
            action_class_recalls, severity_class_recalls
        )
        
        # Check for improvement
        improvement = combined_recall - self.best_combined_recall
        
        if improvement > self.min_delta:
            # Improvement detected
            self.best_combined_recall = combined_recall
            self.patience_counter = 0
            
            # Save best model state
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            
            self.improvement_history.append({
                'epoch': epoch,
                'combined_recall': combined_recall,
                'improvement': improvement,
                'action_macro_recall': action_macro_recall,
                'severity_macro_recall': severity_macro_recall
            })
            
            logger.info(f"New best combined macro recall: {combined_recall:.4f} "
                       f"(+{improvement:.4f}) at epoch {epoch}")
            
        else:
            # No improvement
            self.patience_counter += 1
            logger.info(f"No improvement in combined macro recall. "
                       f"Patience: {self.patience_counter}/{self.patience}")
        
        # Check if should stop
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch}")
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                logger.info(f"Restored best model weights from epoch with "
                           f"combined recall: {self.best_combined_recall:.4f}")
        
        return self.should_stop
    
    def _calculate_weighted_combined_recall(self,
                                          action_macro_recall: float,
                                          severity_macro_recall: float,
                                          action_class_recalls: List[float],
                                          severity_class_recalls: List[float]) -> float:
        """
        Calculate weighted combined macro recall with emphasis on minority classes.
        
        Args:
            action_macro_recall: Action task macro recall
            severity_macro_recall: Severity task macro recall
            action_class_recalls: Per-class recalls for action task
            severity_class_recalls: Per-class recalls for severity task
            
        Returns:
            Weighted combined macro recall
        """
        # Base combined recall (equal weight)
        base_combined = (action_macro_recall + severity_macro_recall) / 2
        
        # Calculate minority class bonus
        minority_bonus = 0.0
        
        # Action minority classes: Pushing (4), Dive (7)
        if len(action_class_recalls) > 7:
            minority_action_recall = (action_class_recalls[4] + action_class_recalls[7]) / 2
            minority_bonus += minority_action_recall * self.minority_class_weight * 0.1
        
        # Severity minority class: Red Card (3)
        if len(severity_class_recalls) > 3:
            minority_severity_recall = severity_class_recalls[3]
            minority_bonus += minority_severity_recall * self.minority_class_weight * 0.1
        
        # Combined score with minority bonus
        weighted_combined = base_combined + minority_bonus
        
        return weighted_combined
    
    def get_stats(self) -> Dict:
        """Get early stopping statistics."""
        return {
            'best_combined_recall': self.best_combined_recall,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop,
            'improvement_history': self.improvement_history[-5:]  # Last 5 improvements
        }


class AdvancedTrainingStrategiesManager:
    """
    Manager class that coordinates all advanced training strategies.
    
    Integrates all components: scheduler, gradient accumulation, and early stopping.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 scheduler_config: Dict = None,
                 accumulation_config: Dict = None,
                 early_stopping_config: Dict = None,
                 save_dir: str = "advanced_training_logs"):
        """
        Initialize the advanced training strategies manager.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_config: Configuration for scheduler
            accumulation_config: Configuration for gradient accumulation
            early_stopping_config: Configuration for early stopping
            save_dir: Directory to save logs and checkpoints
        """
        self.optimizer = optimizer
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components with default configs if not provided
        scheduler_config = scheduler_config or self._get_default_scheduler_config()
        accumulation_config = accumulation_config or self._get_default_accumulation_config()
        early_stopping_config = early_stopping_config or self._get_default_early_stopping_config()
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingWarmRestartsScheduler(
            optimizer=optimizer,
            **scheduler_config
        )
        
        # Initialize gradient accumulator
        self.gradient_accumulator = EnhancedGradientAccumulator(
            **accumulation_config
        )
        
        # Initialize early stopping
        self.early_stopping = CombinedMacroRecallEarlyStopping(
            **early_stopping_config
        )
        
        # Training statistics
        self.training_stats = {
            'epochs_completed': 0,
            'total_restarts': 0,
            'early_stopping_triggered': False,
            'best_combined_recall': -np.inf
        }
        
        logger.info("AdvancedTrainingStrategiesManager initialized successfully!")
    
    def _get_default_scheduler_config(self) -> Dict:
        """Get default scheduler configuration."""
        return {
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6,
            'adaptive_restart': True,
            'minority_performance_threshold': 0.05
        }
    
    def _get_default_accumulation_config(self) -> Dict:
        """Get default gradient accumulation configuration."""
        return {
            'base_accumulation_steps': 4,
            'max_accumulation_steps': 16,
            'memory_threshold': 0.85,
            'adaptive_accumulation': True
        }
    
    def _get_default_early_stopping_config(self) -> Dict:
        """Get default early stopping configuration."""
        return {
            'patience': 15,
            'min_delta': 0.001,
            'restore_best_weights': True,
            'minority_class_weight': 2.0
        }
    
    def step_scheduler(self, epoch: int, metrics: Dict):
        """
        Step the scheduler with performance metrics.
        
        Args:
            epoch: Current epoch
            metrics: Performance metrics
        """
        self.scheduler.step(epoch=epoch, metrics=metrics)
        
        # Update training stats
        self.training_stats['epochs_completed'] = epoch
        self.training_stats['total_restarts'] = self.scheduler.total_restarts
    
    def should_accumulate_gradients(self) -> bool:
        """
        Check if gradients should be accumulated.
        
        Returns:
            True if should accumulate, False if should step optimizer
        """
        return self.gradient_accumulator.should_accumulate()
    
    def optimizer_step_completed(self):
        """Mark that an optimizer step has been completed."""
        self.gradient_accumulator.step_completed()
    
    def check_early_stopping(self,
                           model: nn.Module,
                           action_macro_recall: float,
                           severity_macro_recall: float,
                           action_class_recalls: List[float],
                           severity_class_recalls: List[float],
                           epoch: int) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            model: PyTorch model
            action_macro_recall: Action task macro recall
            severity_macro_recall: Severity task macro recall
            action_class_recalls: Per-class recalls for action task
            severity_class_recalls: Per-class recalls for severity task
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        should_stop = self.early_stopping(
            model, action_macro_recall, severity_macro_recall,
            action_class_recalls, severity_class_recalls, epoch
        )
        
        # Update training stats
        if should_stop:
            self.training_stats['early_stopping_triggered'] = True
        
        self.training_stats['best_combined_recall'] = self.early_stopping.best_combined_recall
        
        return should_stop
    
    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Get the effective batch size considering accumulation."""
        return self.gradient_accumulator.get_effective_batch_size(base_batch_size)
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics from all components."""
        return {
            'training_stats': self.training_stats,
            'scheduler_stats': {
                'current_lr': self.get_current_lr(),
                'epochs_since_restart': self.scheduler.epochs_since_restart,
                'total_restarts': self.scheduler.total_restarts,
                'last_restart_epoch': self.scheduler.last_restart_epoch
            },
            'accumulation_stats': self.gradient_accumulator.get_stats(),
            'early_stopping_stats': self.early_stopping.get_stats()
        }
    
    def save_checkpoint(self, epoch: int, model_state: Dict, optimizer_state: Dict) -> str:
        """
        Save comprehensive checkpoint including all strategy states.
        
        Args:
            epoch: Current epoch
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'gradient_accumulator_stats': self.gradient_accumulator.get_stats(),
            'early_stopping_stats': self.early_stopping.get_stats(),
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"advanced_training_checkpoint_epoch_{epoch}.pth"
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Advanced training checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> int:
        """
        Load comprehensive checkpoint and restore all strategy states.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model
            
        Returns:
            Epoch to resume from
        """
        logger.info(f"Loading advanced training checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training stats
        if 'training_stats' in checkpoint:
            self.training_stats.update(checkpoint['training_stats'])
        
        # Restore early stopping state
        if 'early_stopping_stats' in checkpoint:
            early_stopping_stats = checkpoint['early_stopping_stats']
            self.early_stopping.best_combined_recall = early_stopping_stats.get(
                'best_combined_recall', -np.inf
            )
            self.early_stopping.patience_counter = early_stopping_stats.get(
                'patience_counter', 0
            )
            self.early_stopping.should_stop = early_stopping_stats.get(
                'should_stop', False
            )
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Advanced training checkpoint loaded. Resuming from epoch {epoch + 1}")
        
        return epoch + 1


def create_advanced_training_setup(optimizer: optim.Optimizer,
                                 config: Dict = None) -> AdvancedTrainingStrategiesManager:
    """
    Create a complete advanced training setup with all strategies.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        
    Returns:
        Configured AdvancedTrainingStrategiesManager
    """
    logger.info("Creating advanced training setup...")
    
    # Use default config if none provided
    if config is None:
        config = {
            'scheduler': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6,
                'adaptive_restart': True,
                'minority_performance_threshold': 0.05
            },
            'gradient_accumulation': {
                'base_accumulation_steps': 4,
                'max_accumulation_steps': 16,
                'memory_threshold': 0.85,
                'adaptive_accumulation': True
            },
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.001,
                'restore_best_weights': True,
                'minority_class_weight': 2.0
            }
        }
    
    # Create manager
    manager = AdvancedTrainingStrategiesManager(
        optimizer=optimizer,
        scheduler_config=config.get('scheduler'),
        accumulation_config=config.get('gradient_accumulation'),
        early_stopping_config=config.get('early_stopping')
    )
    
    logger.info("Advanced training setup complete!")
    logger.info(f"  Effective batch size multiplier: {manager.gradient_accumulator.current_accumulation_steps}")
    logger.info(f"  Initial learning rate: {manager.get_current_lr():.6f}")
    logger.info(f"  Early stopping patience: {manager.early_stopping.patience}")
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Test the advanced training strategies
    logger.info("Testing Advanced Training Strategies...")
    
    # Create a dummy model and optimizer for testing
    model = nn.Linear(10, 2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create advanced training manager
    manager = create_advanced_training_setup(optimizer)
    
    # Simulate training loop
    for epoch in range(5):
        logger.info(f"\n--- Epoch {epoch + 1} ---")
        
        # Simulate batch processing
        for batch_idx in range(10):
            # Check if should accumulate gradients
            should_accumulate = manager.should_accumulate_gradients()
            
            if not should_accumulate:
                # Simulate optimizer step
                manager.optimizer_step_completed()
                logger.info(f"Optimizer step at batch {batch_idx + 1}")
        
        # Simulate end of epoch
        fake_metrics = {
            'action_class_recall': [0.8, 0.7, 0.6, 0.5, 0.02, 0.9, 0.8, 0.01],  # Low Pushing(4), Dive(7)
            'severity_class_recall': [0.7, 0.8, 0.6, 0.05],  # Low Red Card(3)
            'action_macro_recall': 0.55,
            'severity_macro_recall': 0.54
        }
        
        # Step scheduler
        manager.step_scheduler(epoch, fake_metrics)
        
        # Check early stopping
        should_stop = manager.check_early_stopping(
            model, 
            fake_metrics['action_macro_recall'],
            fake_metrics['severity_macro_recall'],
            fake_metrics['action_class_recall'],
            fake_metrics['severity_class_recall'],
            epoch
        )
        
        # Print stats
        stats = manager.get_comprehensive_stats()
        logger.info(f"Current LR: {stats['scheduler_stats']['current_lr']:.6f}")
        logger.info(f"Effective batch size: {manager.get_effective_batch_size(8)}")
        logger.info(f"Should stop: {should_stop}")
        
        if should_stop:
            break
    
    logger.info("Advanced training strategies test completed!")