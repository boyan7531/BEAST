"""
Complete Enhanced Training Pipeline Integration

This module integrates all enhanced components into a unified training system:
- Enhanced Data Pipeline with synthetic data generation
- Adaptive Loss System with dynamic parameter adjustment
- Advanced Training Strategies with CosineAnnealingWarmRestarts
- Curriculum Learning System with progressive difficulty
- Performance Monitoring and Alert System
- Model Ensemble System with confidence weighting
- Automated Hyperparameter Optimization

Requirements: All requirements integration, Testing strategy validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from pathlib import Path

# Import all integration modules
from enhanced_pipeline_integration import (
    create_enhanced_training_setup,
    analyze_batch_minority_representation
)
from adaptive_loss_integration import (
    AdaptiveLossWrapper,
    create_adaptive_loss_wrapper
)
from advanced_training_integration import (
    create_enhanced_training_loop,
    extract_performance_metrics_for_strategies
)
from curriculum_learning_integration import (
    create_curriculum_training_integration,
    CurriculumTrainingIntegration
)
from performance_monitor_integration import (
    initialize_performance_monitor,
    update_monitor_from_validation,
    apply_corrective_actions,
    check_target_achievement
)
from ensemble_integration import (
    create_ensemble_training_integration,
    EnsembleTrainingIntegration
)
from hyperparameter_optimization_integration import (
    MVFoulsHyperparameterOptimizer,
    create_hyperparameter_optimization_config
)

# Import base components
from model import MVFoulsModel
from enhanced_model import EnhancedMVFoulsModel
from dataset import MVFoulsDataset
from transform import get_train_transforms, get_val_transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompleteEnhancedTrainingSystem:
    """
    Complete enhanced training system that integrates all components.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 save_dir: str = "enhanced_training_results"):
        """
        Initialize the complete enhanced training system.
        
        Args:
            config: Configuration dictionary for all components
            save_dir: Directory to save all results
        """
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory structure
        self._create_directory_structure()
        
        # Initialize components
        self.enhanced_data_pipeline = None
        self.adaptive_loss_wrapper = None
        self.curriculum_integration = None
        self.performance_monitor = None
        self.ensemble_integration = None
        self.hyperparameter_optimizer = None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Results tracking
        self.training_results = {
            'component_performance': {},
            'integration_metrics': {},
            'final_results': {},
            'component_logs': {}
        }
        
        logger.info(f"CompleteEnhancedTrainingSystem initialized")
        logger.info(f"  Save directory: {save_dir}")
        logger.info(f"  Device: {self.device}")
    
    def _create_directory_structure(self):
        """Create organized directory structure for all components"""
        directories = [
            self.save_dir,
            f"{self.save_dir}/enhanced_pipeline",
            f"{self.save_dir}/adaptive_loss",
            f"{self.save_dir}/advanced_training",
            f"{self.save_dir}/curriculum_learning",
            f"{self.save_dir}/performance_monitoring",
            f"{self.save_dir}/ensemble_system",
            f"{self.save_dir}/hyperparameter_optimization",
            f"{self.save_dir}/integration_logs",
            f"{self.save_dir}/final_results"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Directory structure created successfully")
    
    def setup_enhanced_data_pipeline(self) -> Tuple[Any, Any, Dict]:
        """
        Set up the enhanced data pipeline with synthetic data generation.
        
        Returns:
            Tuple of (train_dataloader, val_dataloader, pipeline_stats)
        """
        logger.info("Setting up enhanced data pipeline...")
        
        pipeline_config = self.config.get('enhanced_pipeline', {})
        
        # Create enhanced training setup
        train_dataloader, val_dataloader, pipeline_stats = create_enhanced_training_setup(
            data_folder=self.config['data']['folder'],
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            start_frame=self.config['data']['start_frame'],
            end_frame=self.config['data']['end_frame'],
            batch_size=self.config['training']['batch_size'],
            model_input_size=tuple(self.config['model']['input_size']),
            num_workers=self.config['training']['num_workers'],
            config=pipeline_config
        )
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Analyze minority representation
        minority_analysis = analyze_batch_minority_representation(
            train_dataloader, num_batches_to_check=10
        )
        
        # Store results
        self.training_results['component_performance']['enhanced_pipeline'] = {
            'pipeline_stats': pipeline_stats,
            'minority_analysis': minority_analysis
        }
        
        logger.info("Enhanced data pipeline setup completed")
        logger.info(f"  Training samples: {len(train_dataloader.dataset)}")
        logger.info(f"  Validation samples: {len(val_dataloader.dataset)}")
        logger.info(f"  Synthetic samples: {pipeline_stats['synthetic_samples_generated']}")
        logger.info(f"  Minority guarantee met: {minority_analysis['minority_guarantee_met']}")
        
        return train_dataloader, val_dataloader, pipeline_stats
    
    def setup_adaptive_loss_system(self) -> AdaptiveLossWrapper:
        """
        Set up the adaptive loss system.
        
        Returns:
            AdaptiveLossWrapper instance
        """
        logger.info("Setting up adaptive loss system...")
        
        loss_config = self.config.get('adaptive_loss', {})
        
        # Create adaptive loss wrapper
        self.adaptive_loss_wrapper = create_adaptive_loss_wrapper(
            use_adaptive=True,
            config=None  # Use default config
        )
        
        logger.info("Adaptive loss system setup completed")
        return self.adaptive_loss_wrapper
    
    def setup_model_and_optimizer(self) -> Tuple[nn.Module, optim.Optimizer]:
        """
        Set up the model and optimizer.
        
        Returns:
            Tuple of (model, optimizer)
        """
        logger.info("Setting up model and optimizer...")
        
        # Create enhanced model
        if self.config['model']['use_enhanced']:
            self.model = EnhancedMVFoulsModel(
                num_action_classes=8,
                num_severity_classes=4,
                aggregation=self.config['model']['aggregation']
            ).to(self.device)
            logger.info("Using EnhancedMVFoulsModel")
        else:
            self.model = MVFoulsModel(
                aggregation=self.config['model']['aggregation']
            ).to(self.device)
            logger.info("Using standard MVFoulsModel")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        logger.info(f"Model and optimizer setup completed")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        
        return self.model, self.optimizer
    
    def setup_curriculum_learning(self) -> Optional[CurriculumTrainingIntegration]:
        """
        Set up curriculum learning integration.
        
        Returns:
            CurriculumTrainingIntegration instance or None
        """
        if not self.config.get('curriculum_learning', {}).get('enabled', False):
            logger.info("Curriculum learning disabled")
            return None
        
        logger.info("Setting up curriculum learning...")
        
        curriculum_config = self.config.get('curriculum_learning', {})
        
        # Create curriculum training integration
        self.curriculum_integration = create_curriculum_training_integration(
            total_epochs=self.config['training']['epochs'],
            base_sampler=None,  # Will be set up with enhanced pipeline
            base_loss_system=self.adaptive_loss_wrapper.loss_system if self.adaptive_loss_wrapper else None,
            training_strategies=None,  # Will be set up later
            config=curriculum_config
        )
        
        logger.info("Curriculum learning setup completed")
        return self.curriculum_integration
    
    def setup_performance_monitoring(self):
        """Set up performance monitoring system"""
        logger.info("Setting up performance monitoring...")
        
        monitoring_config = self.config.get('performance_monitoring', {})
        
        # Initialize performance monitor
        self.performance_monitor = initialize_performance_monitor(
            log_dir=f"{self.save_dir}/performance_monitoring",
            custom_thresholds=monitoring_config.get('thresholds', None)
        )
        
        logger.info("Performance monitoring setup completed")
        return self.performance_monitor
    
    def setup_ensemble_system(self) -> Optional[EnsembleTrainingIntegration]:
        """
        Set up ensemble system integration.
        
        Returns:
            EnsembleTrainingIntegration instance or None
        """
        if not self.config.get('ensemble_system', {}).get('enabled', False):
            logger.info("Ensemble system disabled")
            return None
        
        logger.info("Setting up ensemble system...")
        
        class_names = {
            'action': ['Tackling', 'Standing tackling', 'High leg', 'Holding', 
                      'Pushing', 'Elbowing', 'Challenge', 'Dive'],
            'severity': ['No offence', 'Offence + No card', 'Offence + Yellow card', 'Offence + Red card']
        }
        
        # Create ensemble integration
        self.ensemble_integration = create_ensemble_training_integration(
            class_names=class_names,
            ensemble_config=None,  # Use default config
            device=str(self.device)
        )
        
        logger.info("Ensemble system setup completed")
        return self.ensemble_integration
    
    def setup_hyperparameter_optimization(self) -> Optional[MVFoulsHyperparameterOptimizer]:
        """
        Set up hyperparameter optimization system.
        
        Returns:
            MVFoulsHyperparameterOptimizer instance or None
        """
        if not self.config.get('hyperparameter_optimization', {}).get('enabled', False):
            logger.info("Hyperparameter optimization disabled")
            return None
        
        logger.info("Setting up hyperparameter optimization...")
        
        hp_config = self.config.get('hyperparameter_optimization', {})
        
        # Create hyperparameter optimizer
        self.hyperparameter_optimizer = MVFoulsHyperparameterOptimizer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device=self.device,
            save_dir=f"{self.save_dir}/hyperparameter_optimization",
            optimization_config=create_hyperparameter_optimization_config(**hp_config)
        )
        
        logger.info("Hyperparameter optimization setup completed")
        return self.hyperparameter_optimizer
    
    def run_complete_training(self) -> Dict[str, Any]:
        """
        Run the complete enhanced training pipeline.
        
        Returns:
            Dictionary with comprehensive training results
        """
        logger.info("Starting complete enhanced training pipeline...")
        
        # Setup all components
        logger.info("=== COMPONENT SETUP PHASE ===")
        
        # 1. Enhanced Data Pipeline
        train_dataloader, val_dataloader, pipeline_stats = self.setup_enhanced_data_pipeline()
        
        # 2. Adaptive Loss System
        adaptive_loss_wrapper = self.setup_adaptive_loss_system()
        
        # 3. Model and Optimizer
        model, optimizer = self.setup_model_and_optimizer()
        
        # 4. Performance Monitoring
        performance_monitor = self.setup_performance_monitoring()
        
        # 5. Curriculum Learning (optional)
        curriculum_integration = self.setup_curriculum_learning()
        
        # 6. Ensemble System (optional)
        ensemble_integration = self.setup_ensemble_system()
        
        # 7. Hyperparameter Optimization (optional)
        hyperparameter_optimizer = self.setup_hyperparameter_optimization()
        
        logger.info("=== TRAINING PHASE ===")
        
        # Run hyperparameter optimization if enabled
        if hyperparameter_optimizer:
            logger.info("Running hyperparameter optimization...")
            hp_results = hyperparameter_optimizer.optimize_hyperparameters(
                max_epochs=self.config['hyperparameter_optimization'].get('evaluation_epochs', 5)
            )
            self.training_results['component_performance']['hyperparameter_optimization'] = hp_results
            
            # Update training config with optimized parameters
            if hp_results['best_hyperparameters']:
                self._update_config_with_optimized_params(hp_results['best_hyperparameters'])
        
        # Enhanced training loop
        training_history = self._run_enhanced_training_loop(
            model, optimizer, train_dataloader, val_dataloader,
            adaptive_loss_wrapper, performance_monitor,
            curriculum_integration, ensemble_integration
        )
        
        logger.info("=== EVALUATION PHASE ===")
        
        # Final evaluation and analysis
        final_results = self._run_final_evaluation(
            model, val_dataloader, performance_monitor,
            ensemble_integration
        )
        
        # Compile comprehensive results
        comprehensive_results = self._compile_comprehensive_results(
            training_history, final_results
        )
        
        logger.info("Complete enhanced training pipeline finished!")
        return comprehensive_results
    
    def _run_enhanced_training_loop(self,
                                  model: nn.Module,
                                  optimizer: optim.Optimizer,
                                  train_dataloader,
                                  val_dataloader,
                                  adaptive_loss_wrapper: AdaptiveLossWrapper,
                                  performance_monitor,
                                  curriculum_integration: Optional[CurriculumTrainingIntegration],
                                  ensemble_integration: Optional[EnsembleTrainingIntegration]) -> Dict[str, Any]:
        """Run the enhanced training loop with all components integrated"""
        
        logger.info("Starting enhanced training loop...")
        
        # Initialize training components
        scaler = torch.cuda.amp.GradScaler()
        epochs = self.config['training']['epochs']
        
        # Training history
        training_history = {
            'epoch_metrics': [],
            'component_interactions': [],
            'performance_alerts': [],
            'curriculum_transitions': [],
            'ensemble_updates': [],
            'adaptive_adjustments': []
        }
        
        best_combined_recall = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            logger.info(f"\n=== EPOCH {epoch + 1}/{epochs} ===")
            
            # Setup epoch for curriculum learning
            if curriculum_integration:
                epoch_config = curriculum_integration.setup_epoch(epoch)
                training_history['curriculum_transitions'].append({
                    'epoch': epoch,
                    'config': epoch_config
                })
            
            # Training phase
            model.train()
            train_metrics = self._train_epoch(
                model, optimizer, train_dataloader, adaptive_loss_wrapper,
                scaler, curriculum_integration, epoch
            )
            
            # Validation phase
            model.eval()
            val_metrics = self._validate_epoch(
                model, val_dataloader, adaptive_loss_wrapper
            )
            
            # Update performance monitoring
            performance_results = update_monitor_from_validation(
                performance_monitor, epoch, model, val_dataloader,
                adaptive_loss_wrapper.get_criterion_functions()[0],
                adaptive_loss_wrapper.get_criterion_functions()[1],
                self.device, optimizer
            )
            
            # Update adaptive loss system
            adaptive_loss_wrapper.update_from_epoch_metrics(
                val_metrics['action_predictions'],
                val_metrics['severity_predictions'],
                val_metrics['action_targets'],
                val_metrics['severity_targets']
            )
            
            # Update curriculum learning
            if curriculum_integration:
                curriculum_integration.update_performance_metrics(epoch, val_metrics)
            
            # Update ensemble system
            if ensemble_integration:
                checkpoint_added = ensemble_integration.update_ensemble(
                    model, optimizer, epoch, val_metrics
                )
                if checkpoint_added:
                    training_history['ensemble_updates'].append({
                        'epoch': epoch,
                        'performance': val_metrics['combined_macro_recall']
                    })
            
            # Apply corrective actions if needed
            if performance_results['alerts']:
                corrective_updates = apply_corrective_actions(
                    performance_results['corrective_actions'],
                    optimizer=optimizer
                )
                training_history['performance_alerts'].append({
                    'epoch': epoch,
                    'alerts': [alert.message for alert in performance_results['alerts']],
                    'actions_applied': corrective_updates
                })
            
            # Track best performance
            combined_recall = val_metrics['combined_macro_recall']
            if combined_recall > best_combined_recall:
                best_combined_recall = combined_recall
                epochs_without_improvement = 0
                
                # Save best model
                self._save_checkpoint(model, optimizer, epoch, val_metrics, 'best_model')
            else:
                epochs_without_improvement += 1
            
            # Record epoch metrics
            epoch_record = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'performance_results': {
                    'combined_recall': combined_recall,
                    'action_recall': val_metrics['action_macro_recall'],
                    'severity_recall': val_metrics['severity_macro_recall']
                },
                'component_states': {
                    'adaptive_loss_config': adaptive_loss_wrapper.get_current_config(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
            }
            training_history['epoch_metrics'].append(epoch_record)
            
            # Log progress
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Combined Macro Recall: {combined_recall:.4f}")
            logger.info(f"  Action Macro Recall: {val_metrics['action_macro_recall']:.4f}")
            logger.info(f"  Severity Macro Recall: {val_metrics['severity_macro_recall']:.4f}")
            logger.info(f"  Best Combined Recall: {best_combined_recall:.4f}")
            
            # Early stopping check
            if epochs_without_improvement >= self.config['training'].get('early_stopping_patience', 10):
                logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(model, optimizer, epoch, val_metrics, f'epoch_{epoch + 1}')
        
        # Final training statistics
        training_history['final_stats'] = {
            'total_epochs': epoch + 1,
            'best_combined_recall': best_combined_recall,
            'early_stopping_triggered': epochs_without_improvement >= self.config['training'].get('early_stopping_patience', 10)
        }
        
        logger.info("Enhanced training loop completed!")
        return training_history
    
    def _train_epoch(self,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    train_dataloader,
                    adaptive_loss_wrapper: AdaptiveLossWrapper,
                    scaler: torch.cuda.amp.GradScaler,
                    curriculum_integration: Optional[CurriculumTrainingIntegration],
                    epoch: int) -> Dict[str, Any]:
        """Train for one epoch with all enhancements"""
        
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self.config['training'].get('accumulation_steps', 4)
        
        # Collect predictions for metrics
        all_action_preds = []
        all_severity_preds = []
        all_action_targets = []
        all_severity_targets = []
        
        for batch_idx, (videos, action_labels, severity_labels, action_ids) in enumerate(train_dataloader):
            # Move data to device
            videos = [video.to(self.device) for video in videos]
            action_labels = action_labels.to(self.device)
            severity_labels = severity_labels.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                if hasattr(model, 'forward_with_attention'):
                    action_logits, severity_logits, attention_info = model.forward_with_attention(videos)
                else:
                    action_logits, severity_logits = model(videos)
                    attention_info = None
                
                # Compute loss
                if curriculum_integration:
                    # Use curriculum-aware loss
                    total_loss_batch, loss_components = curriculum_integration.compute_curriculum_loss(
                        action_logits, severity_logits,
                        torch.argmax(action_labels, dim=1),
                        torch.argmax(severity_labels, dim=1)
                    )
                else:
                    # Use adaptive loss
                    total_loss_batch, loss_components = adaptive_loss_wrapper.compute_loss(
                        action_logits, severity_logits,
                        torch.argmax(action_labels, dim=1),
                        torch.argmax(severity_labels, dim=1),
                        attention_info
                    )
                
                # Scale loss for accumulation
                scaled_loss = total_loss_batch / accumulation_steps
            
            # Backward pass
            scaler.scale(scaled_loss).backward()
            
            # Optimizer step with accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Store predictions for metrics
            all_action_preds.append(action_logits.detach().cpu())
            all_severity_preds.append(severity_logits.detach().cpu())
            all_action_targets.append(action_labels.detach().cpu())
            all_severity_targets.append(severity_labels.detach().cpu())
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss_batch.item():.4f}")
        
        # Calculate training metrics
        train_action_preds = torch.cat(all_action_preds, dim=0)
        train_severity_preds = torch.cat(all_severity_preds, dim=0)
        train_action_targets = torch.cat(all_action_targets, dim=0)
        train_severity_targets = torch.cat(all_severity_targets, dim=0)
        
        train_metrics = extract_performance_metrics_for_strategies(
            train_action_preds, train_severity_preds,
            train_action_targets, train_severity_targets
        )
        
        train_metrics['average_loss'] = total_loss / max(num_batches, 1)
        
        return train_metrics
    
    def _validate_epoch(self,
                       model: nn.Module,
                       val_dataloader,
                       adaptive_loss_wrapper: AdaptiveLossWrapper) -> Dict[str, Any]:
        """Validate for one epoch"""
        
        total_loss = 0.0
        num_batches = 0
        
        # Collect predictions for metrics
        all_action_preds = []
        all_severity_preds = []
        all_action_targets = []
        all_severity_targets = []
        
        with torch.no_grad():
            for videos, action_labels, severity_labels, action_ids in val_dataloader:
                # Move data to device
                videos = [video.to(self.device) for video in videos]
                action_labels = action_labels.to(self.device)
                severity_labels = severity_labels.to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    if hasattr(model, 'forward_with_attention'):
                        action_logits, severity_logits, attention_info = model.forward_with_attention(videos)
                    else:
                        action_logits, severity_logits = model(videos)
                        attention_info = None
                    
                    # Compute loss
                    total_loss_batch, loss_components = adaptive_loss_wrapper.compute_loss(
                        action_logits, severity_logits,
                        torch.argmax(action_labels, dim=1),
                        torch.argmax(severity_labels, dim=1),
                        attention_info
                    )
                
                # Accumulate metrics
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                # Store predictions for metrics
                all_action_preds.append(action_logits.detach().cpu())
                all_severity_preds.append(severity_logits.detach().cpu())
                all_action_targets.append(action_labels.detach().cpu())
                all_severity_targets.append(severity_labels.detach().cpu())
        
        # Calculate validation metrics
        val_action_preds = torch.cat(all_action_preds, dim=0)
        val_severity_preds = torch.cat(all_severity_preds, dim=0)
        val_action_targets = torch.cat(all_action_targets, dim=0)
        val_severity_targets = torch.cat(all_severity_targets, dim=0)
        
        val_metrics = extract_performance_metrics_for_strategies(
            val_action_preds, val_severity_preds,
            val_action_targets, val_severity_targets
        )
        
        val_metrics['average_loss'] = total_loss / max(num_batches, 1)
        val_metrics['action_predictions'] = val_action_preds
        val_metrics['severity_predictions'] = val_severity_preds
        val_metrics['action_targets'] = val_action_targets
        val_metrics['severity_targets'] = val_severity_targets
        
        return val_metrics
    
    def _run_final_evaluation(self,
                            model: nn.Module,
                            val_dataloader,
                            performance_monitor,
                            ensemble_integration: Optional[EnsembleTrainingIntegration]) -> Dict[str, Any]:
        """Run comprehensive final evaluation"""
        
        logger.info("Running final evaluation...")
        
        # Check target achievement
        target_status = check_target_achievement(performance_monitor)
        
        # Evaluate ensemble if available
        ensemble_results = {}
        if ensemble_integration and ensemble_integration.ensemble_initialized:
            class_names = {
                'action': ['Tackling', 'Standing tackling', 'High leg', 'Holding', 
                          'Pushing', 'Elbowing', 'Challenge', 'Dive'],
                'severity': ['No offence', 'Offence + No card', 'Offence + Yellow card', 'Offence + Red card']
            }
            ensemble_results = ensemble_integration.evaluate_ensemble(val_dataloader, class_names)
        
        # Generate comprehensive performance report
        performance_report = performance_monitor.generate_performance_report()
        
        final_results = {
            'target_achievement': target_status,
            'ensemble_evaluation': ensemble_results,
            'performance_report': performance_report,
            'final_metrics': performance_monitor.epoch_metrics_history[-1].to_dict() if performance_monitor.epoch_metrics_history else {}
        }
        
        logger.info("Final evaluation completed")
        logger.info(f"  Target achieved: {target_status.get('all_targets_achieved', False)}")
        logger.info(f"  Final combined recall: {target_status.get('main_target', {}).get('current', 0.0):.4f}")
        
        return final_results
    
    def _save_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer,
                        epoch: int,
                        metrics: Dict[str, Any],
                        checkpoint_name: str):
        """Save comprehensive checkpoint"""
        
        checkpoint_path = f"{self.save_dir}/final_results/{checkpoint_name}.pth"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _update_config_with_optimized_params(self, optimized_params: Dict[str, Any]):
        """Update configuration with optimized hyperparameters"""
        
        logger.info("Updating configuration with optimized hyperparameters...")
        
        # Update training parameters
        if 'learning_rate' in optimized_params:
            self.config['training']['learning_rate'] = optimized_params['learning_rate']
        
        if 'weight_decay' in optimized_params:
            self.config['training']['weight_decay'] = optimized_params['weight_decay']
        
        if 'gradient_accumulation_steps' in optimized_params:
            self.config['training']['accumulation_steps'] = optimized_params['gradient_accumulation_steps']
        
        logger.info("Configuration updated with optimized parameters")
    
    def _compile_comprehensive_results(self,
                                     training_history: Dict[str, Any],
                                     final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive results from all components"""
        
        logger.info("Compiling comprehensive results...")
        
        comprehensive_results = {
            'training_history': training_history,
            'final_results': final_results,
            'component_performance': self.training_results['component_performance'],
            'integration_metrics': {
                'total_training_time': datetime.now().isoformat(),
                'components_used': {
                    'enhanced_pipeline': self.enhanced_data_pipeline is not None,
                    'adaptive_loss': self.adaptive_loss_wrapper is not None,
                    'curriculum_learning': self.curriculum_integration is not None,
                    'performance_monitoring': self.performance_monitor is not None,
                    'ensemble_system': self.ensemble_integration is not None,
                    'hyperparameter_optimization': self.hyperparameter_optimizer is not None
                },
                'target_achievement': final_results.get('target_achievement', {}),
                'best_performance': training_history.get('final_stats', {}).get('best_combined_recall', 0.0)
            },
            'config': self.config
        }
        
        # Save comprehensive results
        results_path = f"{self.save_dir}/final_results/comprehensive_results.json"
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive results saved to {results_path}")
        
        return comprehensive_results


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the complete enhanced training system"""
    
    return {
        'data': {
            'folder': 'mvfouls',
            'train_split': 'train',
            'val_split': 'test',
            'start_frame': 63,
            'end_frame': 86
        },
        'model': {
            'use_enhanced': True,
            'aggregation': 'attention',
            'input_size': [224, 224]
        },
        'training': {
            'epochs': 30,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_workers': 0,
            'accumulation_steps': 4,
            'early_stopping_patience': 10
        },
        'enhanced_pipeline': {
            'synthetic_generation': {
                'target_classes': {
                    'action': {4: 50, 7: 50},  # Pushing: 50, Dive: 50
                    'severity': {3: 100}       # Red Card: 100
                },
                'mixup_alpha': 0.4,
                'temporal_alpha': 0.3
            },
            'stratified_sampling': {
                'min_minority_per_batch': 1,
                'minority_threshold': 0.05
            }
        },
        'adaptive_loss': {
            'initial_gamma': 2.0,
            'initial_alpha': 1.0,
            'recall_threshold': 0.1,
            'weight_increase_factor': 1.5
        },
        'curriculum_learning': {
            'enabled': True,
            'easy_stage_epochs': 10,
            'medium_stage_epochs': 8,
            'hard_stage_epochs': 7
        },
        'performance_monitoring': {
            'thresholds': {
                'zero_recall_epochs': 3,
                'performance_drop_threshold': 0.1,
                'target_combined_recall': 0.45
            }
        },
        'ensemble_system': {
            'enabled': True,
            'max_checkpoints': 8,
            'min_performance_threshold': 0.35
        },
        'hyperparameter_optimization': {
            'enabled': False,  # Disabled by default due to computational cost
            'optimization_strategy': 'bayesian_gp',
            'n_calls': 20,
            'evaluation_epochs': 3
        }
    }


def run_complete_enhanced_training(config: Optional[Dict[str, Any]] = None,
                                 save_dir: str = "enhanced_training_results") -> Dict[str, Any]:
    """
    Run the complete enhanced training pipeline.
    
    Args:
        config: Configuration dictionary (uses default if None)
        save_dir: Directory to save results
        
    Returns:
        Comprehensive training results
    """
    
    if config is None:
        config = create_default_config()
    
    # Create and run the complete enhanced training system
    training_system = CompleteEnhancedTrainingSystem(config, save_dir)
    results = training_system.run_complete_training()
    
    return results


if __name__ == "__main__":
    print("Testing Complete Enhanced Training Integration...")
    
    # Create test configuration
    test_config = create_default_config()
    test_config['training']['epochs'] = 2  # Short test run
    test_config['hyperparameter_optimization']['enabled'] = False
    test_config['ensemble_system']['enabled'] = False
    
    try:
        # Run complete enhanced training
        results = run_complete_enhanced_training(
            config=test_config,
            save_dir="test_enhanced_training"
        )
        
        print("✓ Complete enhanced training integration test successful!")
        print(f"✓ Best performance: {results['integration_metrics']['best_performance']:.4f}")
        print(f"✓ Components used: {results['integration_metrics']['components_used']}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()