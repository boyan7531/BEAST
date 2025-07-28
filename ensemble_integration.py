"""
Ensemble Integration Module

This module demonstrates how to integrate the ensemble system with the existing
MVFouls training pipeline, including performance monitoring and adaptive training.

Requirements: 7.4, 5.4
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from model_ensemble_system import EnsembleSystem, EnsembleConfig
from enhanced_model import EnhancedMVFoulsModel
from performance_monitor import PerformanceMonitor
from advanced_training_strategies import AdvancedTrainingStrategiesManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleTrainingIntegration:
    """
    Integration class that combines ensemble system with existing training components.
    """
    
    def __init__(self, 
                 ensemble_config: EnsembleConfig,
                 performance_monitor: PerformanceMonitor,
                 training_strategies: Optional[AdvancedTrainingStrategiesManager] = None,
                 device: str = 'cuda'):
        """
        Initialize ensemble training integration.
        
        Args:
            ensemble_config: Configuration for ensemble system
            performance_monitor: Performance monitoring instance
            training_strategies: Optional advanced training strategies
            device: Device for computations
        """
        self.ensemble_system = EnsembleSystem(ensemble_config, device)
        self.performance_monitor = performance_monitor
        self.training_strategies = training_strategies
        self.device = device
        
        # Tracking variables
        self.ensemble_initialized = False
        self.best_ensemble_recall = 0.0
        self.epochs_since_ensemble_improvement = 0
        
        logger.info("Ensemble training integration initialized")
    
    def should_save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """
        Determine if current model should be saved as checkpoint for ensemble.
        
        Args:
            epoch: Current training epoch
            metrics: Current performance metrics
            
        Returns:
            True if checkpoint should be saved
        """
        combined_recall = metrics.get('combined_macro_recall', 0.0)
        
        # Always save if above minimum threshold
        if combined_recall >= self.ensemble_system.config.min_performance_threshold:
            return True
        
        # Save if significant improvement in minority classes
        action_recalls = metrics.get('action_class_recalls', {})
        severity_recalls = metrics.get('severity_class_recalls', {})
        
        # Check for minority class breakthroughs
        minority_breakthrough = False
        for class_id, recall in action_recalls.items():
            if recall > 0.15 and class_id in [2, 3]:  # Pushing, Dive classes
                minority_breakthrough = True
                break
        
        for class_id, recall in severity_recalls.items():
            if recall > 0.20 and class_id == 3:  # Red Card class
                minority_breakthrough = True
                break
        
        if minority_breakthrough:
            logger.info(f"Saving checkpoint due to minority class breakthrough at epoch {epoch}")
            return True
        
        return False
    
    def update_ensemble(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, Any], 
                       scheduler: Optional[Any] = None) -> bool:
        """
        Update ensemble system with current model if criteria are met.
        
        Args:
            model: Current model
            optimizer: Current optimizer
            epoch: Current epoch
            metrics: Performance metrics
            scheduler: Optional scheduler
            
        Returns:
            True if checkpoint was added to ensemble
        """
        if not self.should_save_checkpoint(epoch, metrics):
            return False
        
        checkpoint_path = self.ensemble_system.add_checkpoint(
            model, optimizer, epoch, metrics, scheduler
        )
        
        if checkpoint_path:
            logger.info(f"Added checkpoint to ensemble: {checkpoint_path}")
            
            # Reinitialize ensemble if we have enough models
            if len(self.ensemble_system.checkpoint_manager.checkpoints) >= 2:
                self.ensemble_system.initialize_ensemble()
                self.ensemble_initialized = True
                logger.info("Ensemble predictor reinitialized with updated checkpoints")
            
            return True
        
        return False
    
    def evaluate_ensemble(self, data_loader, class_names: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate ensemble performance on validation data.
        
        Args:
            data_loader: Validation data loader
            class_names: Dictionary mapping task names to class names
            
        Returns:
            Dictionary with ensemble evaluation metrics
        """
        if not self.ensemble_initialized:
            logger.warning("Ensemble not initialized, cannot evaluate")
            return {}
        
        self.ensemble_system.ensemble_predictor.models[0].eval()  # Set first model to eval mode
        
        all_action_preds = []
        all_severity_preds = []
        all_action_targets = []
        all_severity_targets = []
        all_uncertainties = []
        all_diversity_metrics = []
        
        with torch.no_grad():
            for batch_idx, (videos, action_targets, severity_targets) in enumerate(data_loader):
                # Move to device
                videos = [video.to(self.device) for video in videos]
                action_targets = action_targets.to(self.device)
                severity_targets = severity_targets.to(self.device)
                
                # Get ensemble predictions with details
                ensemble_results = self.ensemble_system.predict(videos, return_details=True)
                
                action_logits = ensemble_results['action_logits']
                severity_logits = ensemble_results['severity_logits']
                
                # Get predictions
                action_preds = torch.argmax(action_logits, dim=1)
                severity_preds = torch.argmax(severity_logits, dim=1)
                
                all_action_preds.append(action_preds.cpu())
                all_severity_preds.append(severity_preds.cpu())
                all_action_targets.append(action_targets.cpu())
                all_severity_targets.append(severity_targets.cpu())
                
                # Collect uncertainty and diversity metrics
                all_uncertainties.append(ensemble_results['ensemble_uncertainty'])
                all_diversity_metrics.append(ensemble_results['diversity_metrics'])
                
                # Limit evaluation for efficiency
                if batch_idx >= 50:  # Evaluate on first 50 batches
                    break
        
        # Concatenate all predictions
        all_action_preds = torch.cat(all_action_preds, dim=0)
        all_severity_preds = torch.cat(all_severity_preds, dim=0)
        all_action_targets = torch.cat(all_action_targets, dim=0)
        all_severity_targets = torch.cat(all_severity_targets, dim=0)
        
        # Calculate metrics using performance monitor
        temp_monitor = PerformanceMonitor(class_names)
        temp_monitor.update_metrics(all_action_preds, all_action_targets, 'action')
        temp_monitor.update_metrics(all_severity_preds, all_severity_targets, 'severity')
        
        current_metrics = temp_monitor.get_current_metrics()
        
        # Calculate average uncertainty and diversity
        avg_uncertainty = {}
        avg_diversity = {}
        
        if all_uncertainties:
            # Average uncertainty metrics
            for key in all_uncertainties[0].keys():
                values = [unc[key].mean().item() for unc in all_uncertainties]
                avg_uncertainty[key] = np.mean(values)
        
        if all_diversity_metrics:
            # Average diversity metrics
            for key in all_diversity_metrics[0].keys():
                values = [div[key] for div in all_diversity_metrics]
                avg_diversity[key] = np.mean(values)
        
        ensemble_metrics = {
            'ensemble_action_macro_recall': current_metrics['action']['macro_recall'],
            'ensemble_severity_macro_recall': current_metrics['severity']['macro_recall'],
            'ensemble_combined_macro_recall': (
                current_metrics['action']['macro_recall'] + 
                current_metrics['severity']['macro_recall']
            ) / 2,
            'ensemble_uncertainty': avg_uncertainty,
            'ensemble_diversity': avg_diversity,
            'num_ensemble_models': len(self.ensemble_system.ensemble_predictor.models),
            'model_weights': self.ensemble_system.ensemble_predictor.model_weights
        }
        
        # Update best ensemble performance
        combined_recall = ensemble_metrics['ensemble_combined_macro_recall']
        if combined_recall > self.best_ensemble_recall:
            self.best_ensemble_recall = combined_recall
            self.epochs_since_ensemble_improvement = 0
            logger.info(f"New best ensemble performance: {combined_recall:.4f}")
        else:
            self.epochs_since_ensemble_improvement += 1
        
        return ensemble_metrics
    
    def get_ensemble_prediction_for_sample(self, video_list: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Get detailed ensemble prediction for a single sample.
        
        Args:
            video_list: List of video tensors for the sample
            
        Returns:
            Dictionary with detailed prediction information
        """
        if not self.ensemble_initialized:
            raise ValueError("Ensemble not initialized")
        
        with torch.no_grad():
            ensemble_results = self.ensemble_system.predict(video_list, return_details=True)
        
        # Convert logits to probabilities
        action_probs = torch.softmax(ensemble_results['action_logits'], dim=-1)
        severity_probs = torch.softmax(ensemble_results['severity_logits'], dim=-1)
        
        # Get individual model predictions
        individual_action_probs = []
        individual_severity_probs = []
        
        for action_logits, severity_logits in zip(
            ensemble_results['individual_predictions']['action_logits'],
            ensemble_results['individual_predictions']['severity_logits']
        ):
            individual_action_probs.append(torch.softmax(action_logits, dim=-1))
            individual_severity_probs.append(torch.softmax(severity_logits, dim=-1))
        
        return {
            'ensemble_action_probabilities': action_probs.cpu().numpy(),
            'ensemble_severity_probabilities': severity_probs.cpu().numpy(),
            'individual_action_probabilities': [p.cpu().numpy() for p in individual_action_probs],
            'individual_severity_probabilities': [p.cpu().numpy() for p in individual_severity_probs],
            'uncertainty_estimates': {k: v.cpu().numpy() for k, v in ensemble_results['ensemble_uncertainty'].items()},
            'diversity_metrics': ensemble_results['diversity_metrics'],
            'model_weights': ensemble_results['model_weights']
        }
    
    def should_use_ensemble_for_training(self, epoch: int, current_metrics: Dict[str, Any]) -> bool:
        """
        Determine if ensemble should be used for training guidance.
        
        Args:
            epoch: Current epoch
            current_metrics: Current single model metrics
            
        Returns:
            True if ensemble should be used for guidance
        """
        if not self.ensemble_initialized:
            return False
        
        # Use ensemble if single model performance is stagnating
        combined_recall = current_metrics.get('combined_macro_recall', 0.0)
        
        # Use ensemble if current model is significantly worse than best ensemble
        if self.best_ensemble_recall > combined_recall + 0.05:
            return True
        
        # Use ensemble if training has stagnated
        if hasattr(self.performance_monitor, 'epochs_without_improvement'):
            if self.performance_monitor.epochs_without_improvement > 5:
                return True
        
        return False
    
    def get_ensemble_guided_loss_weights(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Get loss weights guided by ensemble uncertainty.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with suggested loss weights
        """
        if not self.ensemble_initialized:
            return {}
        
        # Get ensemble statistics
        ensemble_stats = self.ensemble_system.get_ensemble_stats()
        
        # Base weights on ensemble model performance
        suggested_weights = {}
        
        # If ensemble shows high uncertainty in certain classes, increase their weights
        # This is a simplified heuristic - in practice, you'd want more sophisticated logic
        
        # Increase weights for classes where ensemble shows high disagreement
        base_action_weight = 1.0
        base_severity_weight = 1.0
        
        # Adjust based on ensemble diversity (high diversity suggests difficult classes)
        if 'checkpoint_performance_range' in ensemble_stats:
            performance_std = ensemble_stats['checkpoint_performance_range'].get('std_recall', 0.0)
            if performance_std > 0.05:  # High variance in ensemble performance
                base_action_weight *= 1.2
                base_severity_weight *= 1.2
        
        suggested_weights = {
            'action_loss_weight': base_action_weight,
            'severity_loss_weight': base_severity_weight,
            'ensemble_guidance_strength': min(performance_std * 10, 0.5) if 'performance_std' in locals() else 0.1
        }
        
        return suggested_weights
    
    def save_ensemble_analysis(self, save_path: str, epoch: int):
        """
        Save comprehensive ensemble analysis.
        
        Args:
            save_path: Path to save analysis
            epoch: Current epoch
        """
        analysis = {
            'epoch': epoch,
            'ensemble_stats': self.ensemble_system.get_ensemble_stats(),
            'ensemble_initialized': self.ensemble_initialized,
            'best_ensemble_recall': self.best_ensemble_recall,
            'epochs_since_ensemble_improvement': self.epochs_since_ensemble_improvement,
            'checkpoint_details': [
                cp.to_dict() for cp in self.ensemble_system.checkpoint_manager.checkpoints
            ]
        }
        
        import json
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Ensemble analysis saved to {save_path}")


def create_ensemble_training_integration(
    class_names: Dict[str, List[str]],
    ensemble_config: Optional[EnsembleConfig] = None,
    device: str = 'cuda'
) -> EnsembleTrainingIntegration:
    """
    Factory function to create ensemble training integration.
    
    Args:
        class_names: Dictionary mapping task names to class names
        ensemble_config: Optional ensemble configuration
        device: Device for computations
        
    Returns:
        EnsembleTrainingIntegration instance
    """
    if ensemble_config is None:
        ensemble_config = EnsembleConfig(
            max_checkpoints=8,
            min_performance_threshold=0.35,
            diversity_weight=0.2,
            ensemble_voting_method="confidence_weighted",
            save_dir="ensemble_checkpoints"
        )
    
    # Create performance monitor
    action_class_names = {i: name for i, name in enumerate(class_names['action'])}
    severity_class_names = {i: name for i, name in enumerate(class_names['severity'])}
    performance_monitor = PerformanceMonitor(action_class_names, severity_class_names)
    
    # Create ensemble integration
    integration = EnsembleTrainingIntegration(
        ensemble_config=ensemble_config,
        performance_monitor=performance_monitor,
        device=device
    )
    
    return integration


def demonstrate_ensemble_usage():
    """Demonstrate how to use the ensemble system in training."""
    
    print("Demonstrating Ensemble System Usage...")
    
    # Setup
    class_names = {
        'action': ['Tackling', 'Standing tackling', 'High leg', 'Pushing', 'Elbowing', 'Holding', 'Dive', 'Other'],
        'severity': ['No card', 'Yellow card', 'Second yellow card', 'Red card']
    }
    
    device = 'cpu'  # Use CPU for demo
    
    # Create ensemble integration
    integration = create_ensemble_training_integration(class_names, device=device)
    print("✓ Ensemble integration created")
    
    # Simulate training loop
    model = EnhancedMVFoulsModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(1, 6):
        print(f"\n--- Epoch {epoch} ---")
        
        # Simulate metrics (improving over time)
        base_recall = 0.30 + epoch * 0.03
        metrics = {
            'combined_macro_recall': base_recall + np.random.normal(0, 0.02),
            'action_macro_recall': base_recall + np.random.normal(0, 0.03),
            'severity_macro_recall': base_recall + np.random.normal(0, 0.03),
            'action_class_recalls': {
                0: 0.8, 1: 0.7, 2: max(0.0, epoch * 0.05 - 0.1), 3: max(0.0, epoch * 0.03 - 0.05),
                4: 0.6, 5: 0.5, 6: max(0.0, epoch * 0.04 - 0.08), 7: 0.4
            },
            'severity_class_recalls': {
                0: 0.9, 1: 0.6, 2: 0.3, 3: max(0.0, epoch * 0.06 - 0.1)
            },
            'validation_loss': 2.0 - epoch * 0.2
        }
        
        print(f"Combined macro recall: {metrics['combined_macro_recall']:.3f}")
        
        # Update ensemble
        checkpoint_added = integration.update_ensemble(model, optimizer, epoch, metrics)
        print(f"Checkpoint added: {checkpoint_added}")
        
        # Check if ensemble is ready
        if integration.ensemble_initialized:
            print("✓ Ensemble is initialized and ready for predictions")
            
            # Demonstrate ensemble guidance
            use_ensemble = integration.should_use_ensemble_for_training(epoch, metrics)
            print(f"Should use ensemble for guidance: {use_ensemble}")
            
            if use_ensemble:
                loss_weights = integration.get_ensemble_guided_loss_weights(metrics)
                print(f"Ensemble-guided loss weights: {loss_weights}")
        
        # Save analysis
        analysis_path = f"ensemble_analysis_epoch_{epoch}.json"
        integration.save_ensemble_analysis(analysis_path, epoch)
    
    print("\n✓ Ensemble demonstration completed!")
    
    # Final ensemble stats
    if integration.ensemble_initialized:
        final_stats = integration.ensemble_system.get_ensemble_stats()
        print(f"\nFinal ensemble stats:")
        print(f"  Number of checkpoints: {final_stats['num_checkpoints']}")
        print(f"  Number of ensemble models: {final_stats['num_ensemble_models']}")
        print(f"  Performance range: {final_stats['checkpoint_performance_range']}")


if __name__ == "__main__":
    demonstrate_ensemble_usage()