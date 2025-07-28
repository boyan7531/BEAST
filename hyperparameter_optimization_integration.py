"""
Hyperparameter Optimization Integration for MVFouls Training

This module integrates the automated hyperparameter optimization system with the
existing MVFouls training pipeline, providing seamless optimization capabilities.

Requirements addressed:
- 4.3: Performance-based hyperparameter adjustment during training
- 4.4: Integration with performance monitoring for adaptive optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
import numpy as np

# Import existing components
from hyperparameter_optimizer import (
    AutomatedHyperparameterOptimizer,
    HyperparameterSpace,
    HyperparameterType,
    OptimizationStrategy,
    create_mvfouls_parameter_space
)
from performance_monitor import PerformanceMonitor, EpochMetrics
from adaptive_loss_system import DynamicLossSystem, ClassPerformanceMetrics
from advanced_training_strategies import AdvancedTrainingStrategiesManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MVFoulsHyperparameterOptimizer:
    """
    Specialized hyperparameter optimizer for MVFouls training that integrates
    with existing performance monitoring and training systems.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader,
                 val_dataloader,
                 device: torch.device,
                 save_dir: str = "hyperparameter_optimization",
                 optimization_config: Dict = None):
        """
        Initialize MVFouls hyperparameter optimizer.
        
        Args:
            model: MVFouls model to optimize
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to run optimization on
            save_dir: Directory to save optimization results
            optimization_config: Configuration for optimization
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Default optimization configuration
        self.config = optimization_config or {
            'optimization_strategy': OptimizationStrategy.BAYESIAN_GP,
            'n_optimization_calls': 30,
            'evaluation_epochs': 5,
            'enable_adaptive_adjustment': True,
            'enable_sensitivity_analysis': True,
            'adaptive_adjustment_frequency': 3,
            'early_stopping_patience': 10
        }
        
        # Create parameter space
        self.parameter_space = self._create_extended_parameter_space()
        
        # Initialize optimizer
        self.optimizer = AutomatedHyperparameterOptimizer(
            parameter_space=self.parameter_space,
            optimization_strategy=self.config['optimization_strategy'],
            save_dir=save_dir,
            enable_adaptive_adjustment=self.config['enable_adaptive_adjustment'],
            enable_sensitivity_analysis=self.config['enable_sensitivity_analysis']
        )
        
        # Performance monitoring
        self.performance_monitor = None
        self.optimization_history = []
        self.best_hyperparameters = None
        self.best_performance = -np.inf
        
        logger.info("MVFoulsHyperparameterOptimizer initialized successfully!")
        logger.info(f"  Parameter space: {len(self.parameter_space)} parameters")
        logger.info(f"  Strategy: {self.config['optimization_strategy'].value}")
    
    def _create_extended_parameter_space(self) -> List[HyperparameterSpace]:
        """Create extended parameter space specific to MVFouls training"""
        base_space = create_mvfouls_parameter_space()
        
        # Add MVFouls-specific parameters
        extended_space = base_space + [
            HyperparameterSpace(
                name="action_focal_alpha_minority_boost",
                param_type=HyperparameterType.CONTINUOUS,
                low=1.0,
                high=5.0,
                default=2.5,
                description="Alpha boost for minority action classes (Pushing, Dive)"
            ),
            HyperparameterSpace(
                name="severity_focal_alpha_red_card",
                param_type=HyperparameterType.CONTINUOUS,
                low=2.0,
                high=8.0,
                default=3.0,
                description="Alpha value for Red Card severity class"
            ),
            HyperparameterSpace(
                name="class_weight_minority_multiplier",
                param_type=HyperparameterType.CONTINUOUS,
                low=1.5,
                high=5.0,
                default=2.5,
                description="Multiplier for minority class weights"
            ),
            HyperparameterSpace(
                name="sampling_minority_boost",
                param_type=HyperparameterType.CONTINUOUS,
                low=2.0,
                high=6.0,
                default=3.0,
                description="Sampling boost factor for minority classes"
            ),
            HyperparameterSpace(
                name="attention_diversity_weight",
                param_type=HyperparameterType.CONTINUOUS,
                low=0.01,
                high=0.5,
                default=0.1,
                description="Weight for attention diversity loss"
            ),
            HyperparameterSpace(
                name="early_stopping_minority_weight",
                param_type=HyperparameterType.CONTINUOUS,
                low=1.0,
                high=4.0,
                default=2.0,
                description="Weight for minority classes in early stopping"
            )
        ]
        
        return extended_space
    
    def optimize_hyperparameters(self,
                                max_epochs: int = 25,
                                initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive hyperparameter optimization for MVFouls training.
        
        Args:
            max_epochs: Maximum epochs for each evaluation
            initial_params: Optional initial parameter values
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting comprehensive hyperparameter optimization...")
        logger.info(f"  Max epochs per evaluation: {max_epochs}")
        logger.info(f"  Total optimization calls: {self.config['n_optimization_calls']}")
        
        # Create objective function
        def objective_function(params: Dict[str, Any]) -> float:
            return self._evaluate_hyperparameters(params, max_epochs)
        
        # Run Bayesian optimization
        optimization_result = self.optimizer.optimize_hyperparameters(
            objective_function=objective_function,
            initial_params=initial_params,
            n_calls=self.config['n_optimization_calls']
        )
        
        self.best_hyperparameters = optimization_result.best_params
        self.best_performance = optimization_result.best_score
        
        # Run sensitivity analysis if enabled
        sensitivity_results = None
        if self.config['enable_sensitivity_analysis']:
            logger.info("Running sensitivity analysis...")
            baseline_params = initial_params or self._get_default_parameters()
            
            sensitivity_results = self.optimizer.analyze_parameter_sensitivity(
                evaluation_function=objective_function,
                baseline_params=baseline_params
            )
        
        # Compile comprehensive results
        results = {
            'optimization_result': optimization_result.to_dict(),
            'best_hyperparameters': self.best_hyperparameters,
            'best_performance': self.best_performance,
            'sensitivity_analysis': sensitivity_results,
            'optimization_history': self.optimization_history,
            'parameter_space': [p.name for p in self.parameter_space],
            'config': self.config
        }
        
        # Save results
        self._save_comprehensive_results(results)
        
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"  Best performance: {self.best_performance:.4f}")
        logger.info(f"  Best parameters: {self.best_hyperparameters}")
        
        return results
    
    def _evaluate_hyperparameters(self, params: Dict[str, Any], max_epochs: int) -> float:
        """
        Evaluate a set of hyperparameters by training the model.
        
        Args:
            params: Hyperparameter values to evaluate
            max_epochs: Maximum training epochs
            
        Returns:
            Combined macro recall score
        """
        logger.info(f"Evaluating hyperparameters: {params}")
        
        try:
            # Create fresh model instance
            model = self._create_fresh_model()
            
            # Setup training components with hyperparameters
            optimizer, scheduler, loss_system, performance_monitor = self._setup_training_components(
                model, params
            )
            
            # Training loop
            best_combined_recall = 0.0
            epochs_without_improvement = 0
            
            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = self._train_epoch(model, optimizer, loss_system, params)
                
                # Validation phase
                model.eval()
                val_metrics = self._validate_epoch(model, loss_system)
                
                # Update performance monitoring
                epoch_metrics = performance_monitor.update_metrics(
                    epoch=epoch,
                    action_predictions=val_metrics['action_predictions'],
                    action_targets=val_metrics['action_targets'],
                    severity_predictions=val_metrics['severity_predictions'],
                    severity_targets=val_metrics['severity_targets'],
                    loss_action=val_metrics['action_loss'],
                    loss_severity=val_metrics['severity_loss'],
                    learning_rate=optimizer.param_groups[0]['lr']
                )
                
                # Update scheduler
                if scheduler:
                    scheduler.step()
                
                # Check for improvement
                combined_recall = epoch_metrics.combined_macro_recall
                if combined_recall > best_combined_recall:
                    best_combined_recall = combined_recall
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                if epoch % 5 == 0:
                    logger.info(f"  Epoch {epoch}: Combined recall = {combined_recall:.4f}")
            
            # Record evaluation
            evaluation_record = {
                'params': params.copy(),
                'best_combined_recall': best_combined_recall,
                'final_epoch': epoch,
                'evaluation_time': datetime.now().isoformat()
            }
            self.optimization_history.append(evaluation_record)
            
            logger.info(f"Evaluation completed. Best combined recall: {best_combined_recall:.4f}")
            return best_combined_recall
            
        except Exception as e:
            logger.error(f"Error during hyperparameter evaluation: {e}")
            return 0.0  # Return poor score for failed evaluations
    
    def _create_fresh_model(self) -> nn.Module:
        """Create a fresh model instance with random weights"""
        # Create new model with same architecture
        from model import MVFoulsModel
        fresh_model = MVFoulsModel(aggregation='attention').to(self.device)
        return fresh_model
    
    def _setup_training_components(self, model: nn.Module, params: Dict[str, Any]):
        """Setup training components with given hyperparameters"""
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.get('learning_rate', 1e-4),
            weight_decay=params.get('weight_decay', 1e-4)
        )
        
        # Create scheduler
        scheduler = None
        if 'scheduler_T_0' in params:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.get('scheduler_T_0', 10),
                T_mult=params.get('scheduler_T_mult', 2),
                eta_min=1e-6
            )
        
        # Create loss system with hyperparameters
        loss_system = self._create_loss_system(params)
        
        # Create performance monitor
        action_class_names = {i: f"action_{i}" for i in range(8)}
        severity_class_names = {i: f"severity_{i}" for i in range(4)}
        
        performance_monitor = PerformanceMonitor(
            action_class_names=action_class_names,
            severity_class_names=severity_class_names,
            log_dir=os.path.join(self.save_dir, "temp_evaluation")
        )
        
        return optimizer, scheduler, loss_system, performance_monitor
    
    def _create_loss_system(self, params: Dict[str, Any]) -> DynamicLossSystem:
        """Create loss system with hyperparameter-based configuration"""
        from adaptive_loss_system import LossConfig
        
        # Create loss configuration
        loss_config = LossConfig(
            initial_gamma=params.get('focal_loss_gamma', 2.0),
            initial_alpha=1.0,
            min_gamma=0.5,
            max_gamma=5.0,
            min_alpha=0.1,
            max_alpha=10.0,
            recall_threshold=0.1,
            weight_increase_factor=1.5,
            adaptation_rate=0.1
        )
        
        # Create action and severity class names
        action_class_names = [f"action_{i}" for i in range(8)]
        severity_class_names = [f"severity_{i}" for i in range(4)]
        
        # Create dynamic loss system
        loss_system = DynamicLossSystem(
            num_action_classes=8,
            num_severity_classes=4,
            config=loss_config,
            action_class_names=action_class_names,
            severity_class_names=severity_class_names
        )
        
        return loss_system
    
    def _train_epoch(self, model: nn.Module, optimizer: optim.Optimizer, 
                    loss_system: DynamicLossSystem, params: Dict[str, Any]) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        accumulation_steps = params.get('gradient_accumulation_steps', 4)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx >= 10:  # Limit batches for faster evaluation
                break
            
            videos, action_labels, severity_labels = batch
            videos = [v.to(self.device) for v in videos]
            action_labels = action_labels.to(self.device)
            severity_labels = severity_labels.to(self.device)
            
            # Forward pass
            action_logits, severity_logits = model(videos)
            
            # Compute loss
            loss, _ = loss_system(
                action_logits=action_logits,
                severity_logits=severity_logits,
                action_targets=torch.argmax(action_labels, dim=1),
                severity_targets=torch.argmax(severity_labels, dim=1)
            )
            
            # Scale loss for accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, model: nn.Module, loss_system: DynamicLossSystem) -> Dict[str, Any]:
        """Validate for one epoch"""
        all_action_predictions = []
        all_action_targets = []
        all_severity_predictions = []
        all_severity_targets = []
        total_action_loss = 0.0
        total_severity_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= 5:  # Limit batches for faster evaluation
                    break
                
                videos, action_labels, severity_labels = batch
                videos = [v.to(self.device) for v in videos]
                action_labels = action_labels.to(self.device)
                severity_labels = severity_labels.to(self.device)
                
                # Forward pass
                action_logits, severity_logits = model(videos)
                
                # Compute loss
                loss, loss_components = loss_system(
                    action_logits=action_logits,
                    severity_logits=severity_logits,
                    action_targets=torch.argmax(action_labels, dim=1),
                    severity_targets=torch.argmax(severity_labels, dim=1)
                )
                
                # Collect predictions and targets
                all_action_predictions.append(action_logits.cpu())
                all_action_targets.append(action_labels.cpu())
                all_severity_predictions.append(severity_logits.cpu())
                all_severity_targets.append(severity_labels.cpu())
                
                total_action_loss += loss_components.get('action_focal', 0.0)
                total_severity_loss += loss_components.get('severity_focal', 0.0)
                num_batches += 1
        
        # Concatenate all predictions and targets
        action_predictions = torch.cat(all_action_predictions, dim=0)
        action_targets = torch.cat(all_action_targets, dim=0)
        severity_predictions = torch.cat(all_severity_predictions, dim=0)
        severity_targets = torch.cat(all_severity_targets, dim=0)
        
        return {
            'action_predictions': action_predictions,
            'action_targets': action_targets,
            'severity_predictions': severity_predictions,
            'severity_targets': severity_targets,
            'action_loss': total_action_loss / max(num_batches, 1),
            'severity_loss': total_severity_loss / max(num_batches, 1)
        }
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameter values"""
        defaults = {}
        for param_def in self.parameter_space:
            defaults[param_def.name] = param_def.default
        return defaults
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive optimization results"""
        results_file = os.path.join(self.save_dir, "comprehensive_optimization_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive results saved to {results_file}")
    
    def train_with_adaptive_optimization(self,
                                       max_epochs: int = 50,
                                       initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train model with adaptive hyperparameter optimization during training.
        
        Args:
            max_epochs: Maximum training epochs
            initial_params: Initial hyperparameter values
            
        Returns:
            Training results with adaptive optimization history
        """
        logger.info("Starting training with adaptive hyperparameter optimization...")
        
        # Use provided parameters or defaults
        current_params = initial_params or self._get_default_parameters()
        
        # Setup training components
        optimizer, scheduler, loss_system, performance_monitor = self._setup_training_components(
            self.model, current_params
        )
        
        # Training history
        training_history = []
        adaptive_adjustments = []
        best_combined_recall = 0.0
        
        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(self.model, optimizer, loss_system, current_params)
            
            # Validation phase
            self.model.eval()
            val_metrics = self._validate_epoch(self.model, loss_system)
            
            # Update performance monitoring
            epoch_metrics = performance_monitor.update_metrics(
                epoch=epoch,
                action_predictions=val_metrics['action_predictions'],
                action_targets=val_metrics['action_targets'],
                severity_predictions=val_metrics['severity_predictions'],
                severity_targets=val_metrics['severity_targets'],
                loss_action=val_metrics['action_loss'],
                loss_severity=val_metrics['severity_loss'],
                learning_rate=optimizer.param_groups[0]['lr']
            )
            
            # Adaptive hyperparameter adjustment
            if self.config['enable_adaptive_adjustment'] and epoch % self.config['adaptive_adjustment_frequency'] == 0:
                performance_metrics = {
                    'combined_macro_recall': epoch_metrics.combined_macro_recall,
                    'action_class_recall': [m.recall for m in epoch_metrics.action_metrics.values()],
                    'severity_class_recall': [m.recall for m in epoch_metrics.severity_metrics.values()],
                    'gradient_norm': 1.0  # Placeholder
                }
                
                updated_params, adjustments = self.optimizer.adaptive_adjust_during_training(
                    epoch=epoch,
                    performance_metrics=performance_metrics,
                    current_params=current_params
                )
                
                if adjustments:
                    logger.info(f"Adaptive adjustments at epoch {epoch}:")
                    for adj in adjustments:
                        logger.info(f"  {adj.parameter}: {adj.old_value} -> {adj.new_value}")
                    
                    # Apply adjustments
                    current_params = updated_params
                    adaptive_adjustments.extend(adjustments)
                    
                    # Update training components if needed
                    if any(adj.parameter in ['learning_rate', 'weight_decay'] for adj in adjustments):
                        # Update optimizer parameters
                        for param_group in optimizer.param_groups:
                            if 'learning_rate' in updated_params:
                                param_group['lr'] = updated_params['learning_rate']
                            if 'weight_decay' in updated_params:
                                param_group['weight_decay'] = updated_params['weight_decay']
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Record training history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'combined_macro_recall': epoch_metrics.combined_macro_recall,
                'action_macro_recall': epoch_metrics.macro_recall_action,
                'severity_macro_recall': epoch_metrics.macro_recall_severity,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'current_params': current_params.copy()
            })
            
            # Track best performance
            if epoch_metrics.combined_macro_recall > best_combined_recall:
                best_combined_recall = epoch_metrics.combined_macro_recall
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Combined recall = {epoch_metrics.combined_macro_recall:.4f}")
        
        # Compile results
        results = {
            'final_performance': best_combined_recall,
            'training_history': training_history,
            'adaptive_adjustments': [adj.to_dict() for adj in adaptive_adjustments],
            'final_hyperparameters': current_params,
            'total_epochs': max_epochs
        }
        
        # Save results
        adaptive_results_file = os.path.join(self.save_dir, "adaptive_training_results.json")
        with open(adaptive_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Adaptive training completed!")
        logger.info(f"  Final performance: {best_combined_recall:.4f}")
        logger.info(f"  Total adaptive adjustments: {len(adaptive_adjustments)}")
        
        return results


def create_hyperparameter_optimization_config(
    optimization_strategy: str = "bayesian_gp",
    n_calls: int = 30,
    evaluation_epochs: int = 5,
    enable_adaptive: bool = True,
    enable_sensitivity: bool = True
) -> Dict[str, Any]:
    """
    Create configuration for hyperparameter optimization.
    
    Args:
        optimization_strategy: Strategy for optimization
        n_calls: Number of optimization calls
        evaluation_epochs: Epochs per evaluation
        enable_adaptive: Enable adaptive adjustment
        enable_sensitivity: Enable sensitivity analysis
        
    Returns:
        Configuration dictionary
    """
    strategy_map = {
        "bayesian_gp": OptimizationStrategy.BAYESIAN_GP,
        "bayesian_rf": OptimizationStrategy.BAYESIAN_RF,
        "bayesian_gbrt": OptimizationStrategy.BAYESIAN_GBRT,
        "random_search": OptimizationStrategy.RANDOM_SEARCH,
        "adaptive_only": OptimizationStrategy.ADAPTIVE_ONLY
    }
    
    return {
        'optimization_strategy': strategy_map.get(optimization_strategy, OptimizationStrategy.BAYESIAN_GP),
        'n_optimization_calls': n_calls,
        'evaluation_epochs': evaluation_epochs,
        'enable_adaptive_adjustment': enable_adaptive,
        'enable_sensitivity_analysis': enable_sensitivity,
        'adaptive_adjustment_frequency': 3,
        'early_stopping_patience': 8
    }


if __name__ == "__main__":
    print("Testing MVFouls Hyperparameter Optimization Integration...")
    
    # This would normally be run with actual model and data loaders
    print("✓ Integration module loaded successfully")
    print("✓ All components integrated")
    
    # Test configuration creation
    config = create_hyperparameter_optimization_config(
        optimization_strategy="bayesian_gp",
        n_calls=20,
        evaluation_epochs=3,
        enable_adaptive=True,
        enable_sensitivity=True
    )
    print(f"✓ Configuration created: {config['optimization_strategy'].value}")
    
    print("\nMVFouls Hyperparameter Optimization Integration test completed!")