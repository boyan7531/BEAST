"""
Integration module for Curriculum Learning System with MVFouls training pipeline.

This module provides seamless integration of curriculum learning with:
- Enhanced data pipeline
- Advanced training strategies
- Adaptive loss system
- Performance monitoring

Requirements: 3.4, 1.4
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import os
from datetime import datetime

from curriculum_learning_system import (
    CurriculumLearningManager,
    create_curriculum_learning_setup
)
from enhanced_data_pipeline import StratifiedMinorityBatchSampler
from adaptive_loss_system import DynamicLossSystem
from advanced_training_strategies import AdvancedTrainingStrategiesManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumEnhancedBatchSampler:
    """
    Enhanced batch sampler that combines curriculum learning with minority class guarantees.
    """
    
    def __init__(self,
                 action_labels: List[torch.Tensor],
                 severity_labels: List[torch.Tensor],
                 batch_size: int,
                 curriculum_manager: CurriculumLearningManager,
                 min_minority_per_batch: int = 1,
                 minority_threshold: float = 0.05):
        """
        Initialize curriculum-enhanced batch sampler.
        
        Args:
            action_labels: List of one-hot encoded action labels
            severity_labels: List of one-hot encoded severity labels
            batch_size: Size of each batch
            curriculum_manager: Curriculum learning manager
            min_minority_per_batch: Minimum minority samples per batch
            minority_threshold: Frequency threshold for minority classes
        """
        self.batch_size = batch_size
        self.curriculum_manager = curriculum_manager
        self.min_minority_per_batch = min_minority_per_batch
        self.minority_threshold = minority_threshold
        
        # Convert labels to class indices
        self.action_indices = [torch.argmax(label).item() for label in action_labels]
        self.severity_indices = [torch.argmax(label).item() for label in severity_labels]
        
        # Create base stratified sampler
        self.base_sampler = StratifiedMinorityBatchSampler(
            action_labels=action_labels,
            severity_labels=severity_labels,
            batch_size=batch_size,
            min_minority_per_batch=min_minority_per_batch,
            minority_threshold=minority_threshold
        )
        
        # Current epoch for curriculum tracking
        self.current_epoch = 0
        
        logger.info("CurriculumEnhancedBatchSampler initialized")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling"""
        self.current_epoch = epoch
    
    def __iter__(self):
        """Generate batches with curriculum constraints"""
        # Get curriculum configuration for current epoch
        epoch_config = self.curriculum_manager.get_epoch_configuration(self.current_epoch)
        
        # Get allowed classes for current curriculum stage
        allowed_action_classes = set(epoch_config['included_classes']['action'])
        allowed_severity_classes = set(epoch_config['included_classes']['severity'])
        
        # Get sampling weights from curriculum
        action_sampling_weights = epoch_config['sampling_weights']['action']
        severity_sampling_weights = epoch_config['sampling_weights']['severity']
        
        # Create weighted sample pools
        weighted_indices = self._create_weighted_sample_pools(
            allowed_action_classes, allowed_severity_classes,
            action_sampling_weights, severity_sampling_weights
        )
        
        # Generate batches
        batches = self._generate_curriculum_batches(weighted_indices)
        
        return iter(batches)
    
    def _create_weighted_sample_pools(self,
                                    allowed_action_classes: set,
                                    allowed_severity_classes: set,
                                    action_weights: Dict[int, float],
                                    severity_weights: Dict[int, float]) -> List[int]:
        """Create weighted sample pools based on curriculum constraints"""
        weighted_indices = []
        
        for idx in range(len(self.action_indices)):
            action_class = self.action_indices[idx]
            severity_class = self.severity_indices[idx]
            
            # Check if sample is allowed in current curriculum stage
            if (action_class in allowed_action_classes and 
                severity_class in allowed_severity_classes):
                
                # Calculate combined weight
                action_weight = action_weights.get(action_class, 1.0)
                severity_weight = severity_weights.get(severity_class, 1.0)
                combined_weight = (action_weight + severity_weight) / 2
                
                # Add sample multiple times based on weight
                num_copies = max(1, int(combined_weight))
                weighted_indices.extend([idx] * num_copies)
        
        # Shuffle weighted indices
        np.random.shuffle(weighted_indices)
        
        logger.debug(f"Created weighted sample pool with {len(weighted_indices)} samples "
                    f"from {len(self.action_indices)} total samples")
        
        return weighted_indices
    
    def _generate_curriculum_batches(self, weighted_indices: List[int]) -> List[List[int]]:
        """Generate batches from weighted sample pool"""
        batches = []
        
        # Ensure we have enough samples
        if len(weighted_indices) < self.batch_size:
            logger.warning(f"Not enough curriculum-allowed samples ({len(weighted_indices)}) "
                          f"for batch size ({self.batch_size})")
            # Repeat samples if necessary
            while len(weighted_indices) < self.batch_size:
                weighted_indices.extend(weighted_indices[:min(len(weighted_indices), 
                                                            self.batch_size - len(weighted_indices))])
        
        # Create batches
        for i in range(0, len(weighted_indices), self.batch_size):
            batch = weighted_indices[i:i + self.batch_size]
            
            # Ensure batch has minimum size
            if len(batch) >= self.min_minority_per_batch:
                # Pad batch if necessary
                while len(batch) < self.batch_size:
                    batch.append(np.random.choice(weighted_indices))
                
                batches.append(batch[:self.batch_size])
        
        return batches
    
    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.action_indices) // self.batch_size


class CurriculumAwareLossSystem:
    """
    Loss system that adapts to curriculum learning stages.
    """
    
    def __init__(self,
                 base_loss_system: DynamicLossSystem,
                 curriculum_manager: CurriculumLearningManager):
        """
        Initialize curriculum-aware loss system.
        
        Args:
            base_loss_system: Base adaptive loss system
            curriculum_manager: Curriculum learning manager
        """
        self.base_loss_system = base_loss_system
        self.curriculum_manager = curriculum_manager
        
        # Current epoch tracking
        self.current_epoch = 0
        
        logger.info("CurriculumAwareLossSystem initialized")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling"""
        self.current_epoch = epoch
        
        # Get curriculum configuration
        epoch_config = self.curriculum_manager.get_epoch_configuration(epoch)
        
        # Update loss weights based on curriculum stage
        self._update_loss_weights(epoch_config['loss_weights'])
    
    def _update_loss_weights(self, curriculum_weights: Dict[str, Dict[int, float]]):
        """Update loss weights based on curriculum configuration"""
        # Update action task weights
        if 'action' in curriculum_weights:
            action_weights = curriculum_weights['action']
            
            # Convert to tensor format expected by base loss system
            action_weight_tensor = torch.ones(8)  # 8 action classes
            for class_id, weight in action_weights.items():
                if 0 <= class_id < 8:
                    action_weight_tensor[class_id] = weight
            
            # Update base loss system
            if hasattr(self.base_loss_system, 'action_focal_loss'):
                self.base_loss_system.action_focal_loss.update_class_weights(action_weight_tensor)
        
        # Update severity task weights
        if 'severity' in curriculum_weights:
            severity_weights = curriculum_weights['severity']
            
            # Convert to tensor format
            severity_weight_tensor = torch.ones(4)  # 4 severity classes
            for class_id, weight in severity_weights.items():
                if 0 <= class_id < 4:
                    severity_weight_tensor[class_id] = weight
            
            # Update base loss system
            if hasattr(self.base_loss_system, 'severity_focal_loss'):
                self.base_loss_system.severity_focal_loss.update_class_weights(severity_weight_tensor)
    
    def compute_loss(self, 
                    action_logits: torch.Tensor,
                    severity_logits: torch.Tensor,
                    action_targets: torch.Tensor,
                    severity_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute curriculum-aware loss.
        
        Args:
            action_logits: Action predictions
            severity_logits: Severity predictions
            action_targets: Action targets
            severity_targets: Severity targets
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Get curriculum configuration
        epoch_config = self.curriculum_manager.get_epoch_configuration(self.current_epoch)
        
        # Filter predictions and targets based on curriculum constraints
        filtered_data = self._filter_by_curriculum_constraints(
            action_logits, severity_logits, action_targets, severity_targets,
            epoch_config['included_classes']
        )
        
        if filtered_data is None:
            # No valid samples for current curriculum stage
            logger.warning(f"No valid samples for curriculum stage at epoch {self.current_epoch}")
            return torch.tensor(0.0, requires_grad=True), {}
        
        filtered_action_logits, filtered_severity_logits, filtered_action_targets, filtered_severity_targets = filtered_data
        
        # Compute loss using base system
        total_loss, loss_components = self.base_loss_system.compute_loss(
            filtered_action_logits, filtered_severity_logits,
            filtered_action_targets, filtered_severity_targets
        )
        
        # Add curriculum stage information to loss components
        loss_components['curriculum_stage'] = epoch_config['current_stage'].stage_name
        loss_components['curriculum_progress'] = epoch_config['curriculum_stats']['stage_progress']
        
        return total_loss, loss_components
    
    def _filter_by_curriculum_constraints(self,
                                        action_logits: torch.Tensor,
                                        severity_logits: torch.Tensor,
                                        action_targets: torch.Tensor,
                                        severity_targets: torch.Tensor,
                                        included_classes: Dict[str, List[int]]) -> Optional[Tuple]:
        """Filter batch data based on curriculum constraints"""
        # Convert targets to class indices
        action_classes = torch.argmax(action_targets, dim=1)
        severity_classes = torch.argmax(severity_targets, dim=1)
        
        # Create mask for allowed samples
        allowed_action_classes = set(included_classes['action'])
        allowed_severity_classes = set(included_classes['severity'])
        
        mask = torch.zeros(len(action_classes), dtype=torch.bool)
        for i in range(len(action_classes)):
            action_class = action_classes[i].item()
            severity_class = severity_classes[i].item()
            
            if (action_class in allowed_action_classes and 
                severity_class in allowed_severity_classes):
                mask[i] = True
        
        # Check if we have any valid samples
        if not mask.any():
            return None
        
        # Filter data
        filtered_action_logits = action_logits[mask]
        filtered_severity_logits = severity_logits[mask]
        filtered_action_targets = action_targets[mask]
        filtered_severity_targets = severity_targets[mask]
        
        return (filtered_action_logits, filtered_severity_logits,
                filtered_action_targets, filtered_severity_targets)


class CurriculumTrainingIntegration:
    """
    Main integration class that coordinates curriculum learning with all training components.
    """
    
    def __init__(self,
                 total_epochs: int,
                 base_sampler = None,
                 base_loss_system: DynamicLossSystem = None,
                 training_strategies: AdvancedTrainingStrategiesManager = None,
                 config: Dict = None,
                 save_dir: str = "curriculum_training_logs"):
        """
        Initialize curriculum training integration.
        
        Args:
            total_epochs: Total number of training epochs
            base_sampler: Base batch sampler
            base_loss_system: Base adaptive loss system
            training_strategies: Advanced training strategies manager
            config: Configuration dictionary
            save_dir: Directory to save logs
        """
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize curriculum learning manager
        self.curriculum_manager = create_curriculum_learning_setup(
            total_epochs=total_epochs,
            base_sampler=base_sampler,
            config=config.get('curriculum', {}) if config else None
        )
        
        # Initialize curriculum-aware components
        self.curriculum_sampler = None
        if base_sampler is not None:
            # This will be set up when creating enhanced dataloader
            pass
        
        self.curriculum_loss_system = None
        if base_loss_system is not None:
            self.curriculum_loss_system = CurriculumAwareLossSystem(
                base_loss_system=base_loss_system,
                curriculum_manager=self.curriculum_manager
            )
        
        self.training_strategies = training_strategies
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'curriculum_stages': [],
            'stage_transitions': [],
            'performance_by_stage': {},
            'loss_weight_history': [],
            'sampling_weight_history': []
        }
        
        logger.info(f"CurriculumTrainingIntegration initialized:")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Curriculum loss system: {'Enabled' if self.curriculum_loss_system else 'Disabled'}")
        logger.info(f"  Training strategies: {'Enabled' if self.training_strategies else 'Disabled'}")
    
    def create_curriculum_enhanced_dataloader(self,
                                            action_labels: List[torch.Tensor],
                                            severity_labels: List[torch.Tensor],
                                            dataset,
                                            batch_size: int,
                                            num_workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Create curriculum-enhanced dataloader.
        
        Args:
            action_labels: Action labels for sampling
            severity_labels: Severity labels for sampling
            dataset: Dataset to sample from
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            DataLoader with curriculum-enhanced sampling
        """
        logger.info("Creating curriculum-enhanced dataloader...")
        
        # Create curriculum-enhanced batch sampler
        self.curriculum_sampler = CurriculumEnhancedBatchSampler(
            action_labels=action_labels,
            severity_labels=severity_labels,
            batch_size=batch_size,
            curriculum_manager=self.curriculum_manager
        )
        
        # Create dataloader
        from dataset import custom_collate_fn
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=self.curriculum_sampler,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info("Curriculum-enhanced dataloader created successfully")
        return dataloader
    
    def setup_epoch(self, epoch: int) -> Dict:
        """
        Set up all components for current epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Epoch configuration dictionary
        """
        self.current_epoch = epoch
        
        # Update curriculum sampler
        if self.curriculum_sampler:
            self.curriculum_sampler.set_epoch(epoch)
        
        # Update curriculum loss system
        if self.curriculum_loss_system:
            self.curriculum_loss_system.set_epoch(epoch)
        
        # Get epoch configuration
        epoch_config = self.curriculum_manager.get_epoch_configuration(epoch)
        
        # Log stage information
        stage_name = epoch_config['current_stage'].stage_name
        stage_progress = epoch_config['curriculum_stats']['stage_progress']
        
        logger.info(f"Epoch {epoch}: Curriculum stage '{stage_name}' "
                   f"(Progress: {stage_progress:.2f})")
        
        # Log included classes
        included_actions = epoch_config['included_classes']['action']
        included_severities = epoch_config['included_classes']['severity']
        logger.info(f"  Included action classes: {included_actions}")
        logger.info(f"  Included severity classes: {included_severities}")
        
        # Track stage transitions
        if (len(self.training_history['curriculum_stages']) == 0 or
            self.training_history['curriculum_stages'][-1] != epoch_config['current_stage'].stage_name):
            
            self.training_history['curriculum_stages'].append(epoch_config['current_stage'].stage_name)
            self.training_history['stage_transitions'].append({
                'epoch': epoch,
                'stage': epoch_config['current_stage'].stage_name,
                'progress': stage_progress
            })
        
        # Track weight histories
        self.training_history['loss_weight_history'].append({
            'epoch': epoch,
            'weights': epoch_config['loss_weights']
        })
        self.training_history['sampling_weight_history'].append({
            'epoch': epoch,
            'weights': epoch_config['sampling_weights']
        })
        
        return epoch_config
    
    def compute_curriculum_loss(self,
                              action_logits: torch.Tensor,
                              severity_logits: torch.Tensor,
                              action_targets: torch.Tensor,
                              severity_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute curriculum-aware loss.
        
        Args:
            action_logits: Action predictions
            severity_logits: Severity predictions
            action_targets: Action targets
            severity_targets: Severity targets
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        if self.curriculum_loss_system:
            return self.curriculum_loss_system.compute_loss(
                action_logits, severity_logits, action_targets, severity_targets
            )
        else:
            logger.warning("No curriculum loss system available")
            return torch.tensor(0.0, requires_grad=True), {}
    
    def update_performance_metrics(self, epoch: int, metrics: Dict):
        """
        Update performance metrics for curriculum tracking.
        
        Args:
            epoch: Current epoch
            metrics: Performance metrics dictionary
        """
        # Update curriculum manager
        self.curriculum_manager.update_performance_metrics(epoch, metrics)
        
        # Track performance by stage
        current_stage = self.curriculum_manager.curriculum_scheduler.progress.current_stage
        stage_name = self.curriculum_manager.curriculum_scheduler.stages[current_stage].stage_name
        
        if stage_name not in self.training_history['performance_by_stage']:
            self.training_history['performance_by_stage'][stage_name] = []
        
        self.training_history['performance_by_stage'][stage_name].append({
            'epoch': epoch,
            'metrics': metrics.copy()
        })
        
        # Update training strategies if available
        if self.training_strategies:
            # Check if curriculum learning should trigger early stopping adjustments
            self._check_curriculum_early_stopping_adjustments(epoch, metrics)
    
    def _check_curriculum_early_stopping_adjustments(self, epoch: int, metrics: Dict):
        """Check if curriculum progress should adjust early stopping parameters"""
        # Get curriculum progress
        curriculum_stats = self.curriculum_manager.curriculum_scheduler.get_curriculum_stats()
        
        # If we're in the final stage and performance is improving, extend patience
        if (curriculum_stats['current_stage'] == len(self.curriculum_manager.curriculum_scheduler.stages) - 1 and
            curriculum_stats['stage_progress'] > 0.5):
            
            # Check if minority classes are improving
            minority_improving = False
            if 'action_class_recall' in metrics and 'severity_class_recall' in metrics:
                action_recalls = metrics['action_class_recall']
                severity_recalls = metrics['severity_class_recall']
                
                # Check minority class performance (Pushing=4, Dive=7, Red Card=3)
                if (len(action_recalls) > 7 and len(severity_recalls) > 3):
                    pushing_recall = action_recalls[4]
                    dive_recall = action_recalls[7]
                    red_card_recall = severity_recalls[3]
                    
                    # If any minority class is showing improvement, extend patience
                    if pushing_recall > 0.1 or dive_recall > 0.05 or red_card_recall > 0.15:
                        minority_improving = True
            
            if minority_improving and hasattr(self.training_strategies, 'early_stopping'):
                # Reset patience counter to give more time for minority class learning
                original_patience = self.training_strategies.early_stopping.patience_counter
                self.training_strategies.early_stopping.patience_counter = max(0, original_patience - 2)
                
                logger.info(f"Curriculum learning: Extended early stopping patience due to minority class improvement")
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive training report with curriculum analysis"""
        logger.info("Generating comprehensive curriculum training report...")
        
        # Generate curriculum progress report
        viz_path, logs_path, curriculum_summary = self.curriculum_manager.generate_progress_report()
        
        # Analyze stage-wise performance
        stage_analysis = self._analyze_stage_performance()
        
        # Generate integration statistics
        integration_stats = {
            'total_epochs_completed': self.current_epoch,
            'curriculum_stages_completed': len(self.training_history['stage_transitions']),
            'stage_transitions': self.training_history['stage_transitions'],
            'final_stage': curriculum_summary.get('final_stage_name', 'Unknown'),
            'components_used': {
                'curriculum_sampler': self.curriculum_sampler is not None,
                'curriculum_loss_system': self.curriculum_loss_system is not None,
                'training_strategies': self.training_strategies is not None
            }
        }
        
        # Combine all reports
        comprehensive_report = {
            'curriculum_summary': curriculum_summary,
            'stage_analysis': stage_analysis,
            'integration_stats': integration_stats,
            'training_history': self.training_history,
            'visualization_path': viz_path,
            'detailed_logs_path': logs_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save comprehensive report
        report_path = os.path.join(self.save_dir, 'comprehensive_curriculum_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        logger.info(f"Comprehensive curriculum training report saved to {report_path}")
        
        return comprehensive_report
    
    def _analyze_stage_performance(self) -> Dict:
        """Analyze performance improvements across curriculum stages"""
        stage_analysis = {}
        
        for stage_name, performance_history in self.training_history['performance_by_stage'].items():
            if not performance_history:
                continue
            
            # Calculate performance trends within stage
            epochs = [entry['epoch'] for entry in performance_history]
            combined_recalls = [entry['metrics'].get('combined_macro_recall', 0) for entry in performance_history]
            
            if len(combined_recalls) > 1:
                # Calculate improvement within stage
                initial_recall = combined_recalls[0]
                final_recall = combined_recalls[-1]
                improvement = final_recall - initial_recall
                
                # Calculate average performance
                avg_recall = np.mean(combined_recalls)
                
                stage_analysis[stage_name] = {
                    'epochs_in_stage': len(performance_history),
                    'epoch_range': (min(epochs), max(epochs)),
                    'initial_recall': initial_recall,
                    'final_recall': final_recall,
                    'improvement': improvement,
                    'avg_recall': avg_recall,
                    'performance_trend': 'improving' if improvement > 0.01 else 'stable' if abs(improvement) <= 0.01 else 'declining'
                }
        
        return stage_analysis
    
    def save_checkpoint(self, epoch: int, model_state: Dict, optimizer_state: Dict) -> str:
        """
        Save comprehensive checkpoint including curriculum state.
        
        Args:
            epoch: Current epoch
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            
        Returns:
            Path to saved checkpoint
        """
        # Save curriculum checkpoint
        curriculum_checkpoint_path = self.curriculum_manager.save_checkpoint(epoch)
        
        # Create comprehensive checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'curriculum_checkpoint_path': curriculum_checkpoint_path,
            'training_history': self.training_history,
            'curriculum_stats': self.curriculum_manager.get_comprehensive_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add training strategies state if available
        if self.training_strategies:
            checkpoint_data['training_strategies_stats'] = self.training_strategies.get_comprehensive_stats()
        
        checkpoint_path = os.path.join(
            self.save_dir,
            f"curriculum_training_checkpoint_epoch_{epoch}.pth"
        )
        
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Comprehensive curriculum training checkpoint saved: {checkpoint_path}")
        return checkpoint_path


def create_curriculum_training_integration(total_epochs: int,
                                         base_sampler = None,
                                         base_loss_system: DynamicLossSystem = None,
                                         training_strategies: AdvancedTrainingStrategiesManager = None,
                                         config: Dict = None) -> CurriculumTrainingIntegration:
    """
    Create complete curriculum training integration.
    
    Args:
        total_epochs: Total number of training epochs
        base_sampler: Base batch sampler (optional)
        base_loss_system: Base adaptive loss system (optional)
        training_strategies: Advanced training strategies manager (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Configured CurriculumTrainingIntegration
    """
    logger.info("Creating curriculum training integration...")
    
    integration = CurriculumTrainingIntegration(
        total_epochs=total_epochs,
        base_sampler=base_sampler,
        base_loss_system=base_loss_system,
        training_strategies=training_strategies,
        config=config
    )
    
    logger.info("Curriculum training integration created successfully!")
    return integration


def integrate_curriculum_with_existing_training(train_script_globals: Dict,
                                              total_epochs: int,
                                              config: Dict = None) -> CurriculumTrainingIntegration:
    """
    Integrate curriculum learning with existing training script.
    
    Args:
        train_script_globals: Global variables from training script
        total_epochs: Total number of training epochs
        config: Configuration dictionary (optional)
        
    Returns:
        Configured CurriculumTrainingIntegration
    """
    logger.info("Integrating curriculum learning with existing training script...")
    
    # Extract components from training script
    base_loss_system = train_script_globals.get('adaptive_loss_system')
    training_strategies = train_script_globals.get('strategies_manager')
    
    # Create integration
    integration = create_curriculum_training_integration(
        total_epochs=total_epochs,
        base_loss_system=base_loss_system,
        training_strategies=training_strategies,
        config=config
    )
    
    # Add integration to training script globals
    train_script_globals['curriculum_integration'] = integration
    
    # Add helper functions
    train_script_globals['setup_curriculum_epoch'] = integration.setup_epoch
    train_script_globals['compute_curriculum_loss'] = integration.compute_curriculum_loss
    train_script_globals['update_curriculum_metrics'] = integration.update_performance_metrics
    
    logger.info("Curriculum learning integration with existing training complete!")
    
    return integration


# Example usage and testing
if __name__ == "__main__":
    # Test curriculum training integration
    logger.info("Testing Curriculum Training Integration...")
    
    # Create mock components for testing
    total_epochs = 20
    
    # Create integration
    integration = create_curriculum_training_integration(
        total_epochs=total_epochs
    )
    
    # Simulate training epochs
    for epoch in range(total_epochs):
        # Setup epoch
        epoch_config = integration.setup_epoch(epoch)
        
        # Simulate performance metrics
        fake_metrics = {
            'combined_macro_recall': 0.2 + 0.3 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'action_macro_recall': 0.25 + 0.25 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'severity_macro_recall': 0.15 + 0.35 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'action_class_recall': [0.8, 0.7, 0.6, 0.5, 0.1 + 0.2 * (epoch / total_epochs), 0.6, 0.5, 0.05 + 0.15 * (epoch / total_epochs)],
            'severity_class_recall': [0.9, 0.8, 0.6, 0.05 + 0.25 * (epoch / total_epochs)]
        }
        
        # Update metrics
        integration.update_performance_metrics(epoch, fake_metrics)
        
        logger.info(f"Epoch {epoch}: Stage '{epoch_config['current_stage'].stage_name}' completed")
    
    # Generate final report
    comprehensive_report = integration.generate_comprehensive_report()
    
    logger.info("Curriculum training integration test completed!")
    logger.info(f"Final performance: {comprehensive_report['curriculum_summary'].get('final_performance', {})}")