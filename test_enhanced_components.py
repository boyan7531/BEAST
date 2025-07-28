"""
Comprehensive Unit Tests for Enhanced Training Components

This module provides unit tests for all enhanced training components to ensure
proper functionality and integration.

Requirements: Testing strategy validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import components to test
from enhanced_data_pipeline import (
    StratifiedMinorityBatchSampler,
    MixupVideoGenerator,
    AdvancedVideoAugmentation,
    SyntheticQualityValidator
)
from adaptive_loss_system import (
    DynamicLossSystem,
    AdaptiveFocalLoss,
    ClassBalanceLoss,
    ClassPerformanceMetrics
)
from advanced_training_strategies import (
    AdvancedTrainingStrategiesManager,
    CosineAnnealingWarmRestartsWithAdaptiveRestart,
    GradientAccumulator,
    EarlyStoppingWithMinorityFocus
)
from curriculum_learning_system import (
    CurriculumLearningManager,
    CurriculumScheduler,
    CurriculumStage
)
from performance_monitor import (
    PerformanceMonitor,
    EpochMetrics,
    ClassMetrics,
    PerformanceAlert
)
from model_ensemble_system import (
    EnsembleSystem,
    EnsemblePredictor,
    CheckpointManager
)
from hyperparameter_optimizer import (
    AutomatedHyperparameterOptimizer,
    HyperparameterSpace,
    HyperparameterType
)

# Import integration modules
from enhanced_pipeline_integration import EnhancedDataPipelineManager
from adaptive_loss_integration import AdaptiveLossWrapper
from complete_training_integration import CompleteEnhancedTrainingSystem


class TestEnhancedDataPipeline(unittest.TestCase):
    """Test cases for Enhanced Data Pipeline components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.num_samples = 20
        
        # Create mock labels
        self.action_labels = [torch.randint(0, 8, (1,)) for _ in range(self.num_samples)]
        self.severity_labels = [torch.randint(0, 4, (1,)) for _ in range(self.num_samples)]
        
        # Create minority samples (classes 4, 7 for action, 3 for severity)
        self.action_labels[0] = torch.tensor([4])  # Pushing
        self.action_labels[1] = torch.tensor([7])  # Dive
        self.severity_labels[0] = torch.tensor([3])  # Red Card
    
    def test_stratified_minority_batch_sampler(self):
        """Test StratifiedMinorityBatchSampler functionality"""
        
        # Convert to one-hot for sampler
        action_labels_onehot = [torch.zeros(8).scatter_(0, label, 1) for label in self.action_labels]
        severity_labels_onehot = [torch.zeros(4).scatter_(0, label, 1) for label in self.severity_labels]
        
        sampler = StratifiedMinorityBatchSampler(
            action_labels=action_labels_onehot,
            severity_labels=severity_labels_onehot,
            batch_size=self.batch_size,
            min_minority_per_batch=1
        )
        
        # Test sampler properties
        self.assertEqual(len(sampler), self.num_samples // self.batch_size)
        
        # Test minority stats
        stats = sampler.get_minority_stats()
        self.assertIn('minority_action_classes', stats)
        self.assertIn('minority_severity_classes', stats)
        
        # Test batch generation
        batches = list(iter(sampler))
        self.assertGreater(len(batches), 0)
        
        # Check that batches have correct size
        for batch in batches:
            self.assertLessEqual(len(batch), self.batch_size)
    
    def test_mixup_video_generator(self):
        """Test MixupVideoGenerator functionality"""
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=self.num_samples)
        mock_dataset.__getitem__ = Mock(side_effect=self._mock_dataset_getitem)
        
        target_classes = {
            'action': {4: 5, 7: 5},
            'severity': {3: 10}
        }
        
        generator = MixupVideoGenerator(
            dataset=mock_dataset,
            target_classes=target_classes,
            mixup_alpha=0.4,
            temporal_alpha=0.3
        )
        
        # Test synthetic sample generation
        synthetic_samples = generator.generate_synthetic_samples()
        
        # Check that samples were generated
        self.assertGreater(len(synthetic_samples), 0)
        
        # Check sample structure
        for sample in synthetic_samples[:3]:  # Check first 3 samples
            video, action_label, severity_label, metadata = sample
            self.assertIsInstance(video, torch.Tensor)
            self.assertIsInstance(action_label, int)
            self.assertIsInstance(severity_label, int)
            self.assertIsInstance(metadata, dict)
    
    def _mock_dataset_getitem(self, idx):
        """Mock dataset __getitem__ method"""
        video = torch.randn(8, 3, 224, 224)  # 8 frames, 3 channels, 224x224
        action_label = self.action_labels[idx % len(self.action_labels)].item()
        severity_label = self.severity_labels[idx % len(self.severity_labels)].item()
        action_id = f"action_{idx}"
        return video, action_label, severity_label, action_id
    
    def test_advanced_video_augmentation(self):
        """Test AdvancedVideoAugmentation functionality"""
        
        augmentation = AdvancedVideoAugmentation(
            temporal_jitter_range=(0.8, 1.2),
            spatial_jitter_prob=0.5,
            color_jitter_prob=0.3
        )
        
        # Test video augmentation
        video = torch.randn(8, 3, 224, 224)
        augmented_video = augmentation(video)
        
        # Check output shape
        self.assertEqual(augmented_video.shape[1:], video.shape[1:])  # Same except possibly time dimension
        self.assertIsInstance(augmented_video, torch.Tensor)
    
    def test_synthetic_quality_validator(self):
        """Test SyntheticQualityValidator functionality"""
        
        validator = SyntheticQualityValidator(
            feature_similarity_threshold=0.7,
            temporal_consistency_threshold=0.8,
            label_consistency_threshold=0.9
        )
        
        # Create mock synthetic samples
        synthetic_samples = []
        for i in range(5):
            video = torch.randn(8, 3, 224, 224)
            action_label = 4  # Pushing
            severity_label = 3  # Red Card
            metadata = {'generation_method': 'mixup', 'source_samples': [0, 1]}
            synthetic_samples.append((video, action_label, severity_label, metadata))
        
        # Test validation
        validation_stats = validator.validate_batch(synthetic_samples)
        
        # Check validation results
        self.assertIn('pass_rate', validation_stats)
        self.assertIn('avg_quality_score', validation_stats)
        self.assertIsInstance(validation_stats['pass_rate'], float)
        self.assertIsInstance(validation_stats['avg_quality_score'], float)


class TestAdaptiveLossSystem(unittest.TestCase):
    """Test cases for Adaptive Loss System components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 8
        self.num_severity_classes = 4
        self.batch_size = 4
        
        # Create mock performance metrics
        self.action_metrics = {}
        self.severity_metrics = {}
        
        for i in range(self.num_action_classes):
            self.action_metrics[i] = ClassPerformanceMetrics(
                class_id=i,
                recall=0.5 if i not in [4, 7] else 0.0,  # Zero recall for Pushing, Dive
                precision=0.6,
                f1_score=0.55,
                support=10,
                epochs_since_improvement=2 if i in [4, 7] else 0,
                best_recall=0.5 if i not in [4, 7] else 0.0,
                recall_history=[0.4, 0.5] if i not in [4, 7] else [0.0, 0.0]
            )
        
        for i in range(self.num_severity_classes):
            self.severity_metrics[i] = ClassPerformanceMetrics(
                class_id=i,
                recall=0.4 if i != 3 else 0.0,  # Zero recall for Red Card
                precision=0.5,
                f1_score=0.45,
                support=8,
                epochs_since_improvement=3 if i == 3 else 0,
                best_recall=0.4 if i != 3 else 0.0,
                recall_history=[0.3, 0.4] if i != 3 else [0.0, 0.0]
            )
    
    def test_adaptive_focal_loss(self):
        """Test AdaptiveFocalLoss functionality"""
        
        focal_loss = AdaptiveFocalLoss(
            num_classes=self.num_action_classes,
            class_names=[f"action_{i}" for i in range(self.num_action_classes)]
        )
        
        # Test forward pass
        logits = torch.randn(self.batch_size, self.num_action_classes, requires_grad=True)
        targets = torch.randint(0, self.num_action_classes, (self.batch_size,))
        
        loss = focal_loss(logits, targets)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)
        
        # Test backward pass
        loss.backward()
        self.assertIsNotNone(logits.grad)
        
        # Test parameter updates
        focal_loss.update_parameters(self.action_metrics)
        
        # Check that parameters were updated for poor-performing classes
        current_params = focal_loss.get_current_parameters()
        self.assertIn('gamma_per_class', current_params)
        self.assertIn('alpha_per_class', current_params)
    
    def test_dynamic_loss_system(self):
        """Test DynamicLossSystem functionality"""
        
        loss_system = DynamicLossSystem(
            num_action_classes=self.num_action_classes,
            num_severity_classes=self.num_severity_classes,
            action_class_names=[f"action_{i}" for i in range(self.num_action_classes)],
            severity_class_names=[f"severity_{i}" for i in range(self.num_severity_classes)]
        )
        
        # Test forward pass
        action_logits = torch.randn(self.batch_size, self.num_action_classes, requires_grad=True)
        severity_logits = torch.randn(self.batch_size, self.num_severity_classes, requires_grad=True)
        action_targets = torch.randint(0, self.num_action_classes, (self.batch_size,))
        severity_targets = torch.randint(0, self.num_severity_classes, (self.batch_size,))
        
        total_loss, loss_components = loss_system(
            action_logits, severity_logits, action_targets, severity_targets
        )
        
        # Check loss properties
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_components, dict)
        self.assertIn('action_focal', loss_components)
        self.assertIn('severity_focal', loss_components)
        self.assertIn('total', loss_components)
        
        # Test backward pass
        total_loss.backward()
        self.assertIsNotNone(action_logits.grad)
        self.assertIsNotNone(severity_logits.grad)
        
        # Test metrics update
        loss_system.update_from_metrics(self.action_metrics, self.severity_metrics)
        
        # Check configuration update
        config = loss_system.get_current_config()
        self.assertIsInstance(config, dict)


class TestAdvancedTrainingStrategies(unittest.TestCase):
    """Test cases for Advanced Training Strategies components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = nn.Linear(10, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
    
    def test_cosine_annealing_warm_restarts_adaptive(self):
        """Test CosineAnnealingWarmRestartsWithAdaptiveRestart"""
        
        scheduler = CosineAnnealingWarmRestartsWithAdaptiveRestart(
            optimizer=self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6,
            adaptive_restart=True,
            minority_performance_threshold=0.05
        )
        
        # Test initial state
        self.assertEqual(scheduler.T_0, 5)
        self.assertEqual(scheduler.T_mult, 2)
        self.assertTrue(scheduler.adaptive_restart)
        
        # Test scheduler step
        initial_lr = self.optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # LR should change after step
        self.assertNotEqual(self.optimizer.param_groups[0]['lr'], initial_lr)
        
        # Test adaptive restart
        minority_recalls = [0.02, 0.01, 0.03]  # Below threshold
        should_restart = scheduler.should_trigger_adaptive_restart(minority_recalls)
        self.assertTrue(should_restart)
        
        # Test restart
        scheduler.trigger_restart()
        self.assertEqual(scheduler.T_cur, 0)
    
    def test_gradient_accumulator(self):
        """Test GradientAccumulator functionality"""
        
        accumulator = GradientAccumulator(
            base_accumulation_steps=4,
            max_accumulation_steps=16,
            memory_threshold=0.8,
            adaptive_accumulation=True
        )
        
        # Test initial state
        self.assertEqual(accumulator.current_accumulation_steps, 4)
        self.assertEqual(accumulator.steps_since_last_update, 0)
        
        # Test should accumulate
        should_accumulate = accumulator.should_accumulate_gradients()
        self.assertTrue(should_accumulate)
        
        # Test step completion
        accumulator.optimizer_step_completed()
        self.assertEqual(accumulator.steps_since_last_update, 0)
        
        # Test adaptive adjustment
        accumulator.adjust_accumulation_steps(memory_usage=0.9, gradient_norm=5.0)
        # Should increase accumulation steps due to high memory usage
        self.assertGreaterEqual(accumulator.current_accumulation_steps, 4)
    
    def test_early_stopping_with_minority_focus(self):
        """Test EarlyStoppingWithMinorityFocus functionality"""
        
        early_stopping = EarlyStoppingWithMinorityFocus(
            patience=5,
            min_delta=0.01,
            restore_best_weights=True,
            minority_class_weight=2.0
        )
        
        # Test initial state
        self.assertEqual(early_stopping.patience, 5)
        self.assertEqual(early_stopping.patience_counter, 0)
        self.assertFalse(early_stopping.early_stop)
        
        # Test improvement detection
        action_recalls = [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4]  # Classes 4,5,6,7 are minority
        severity_recalls = [0.6, 0.7, 0.8, 0.1]  # Class 3 is minority
        
        should_stop = early_stopping.check_early_stopping(
            self.model, 0.65, 0.55, action_recalls, severity_recalls, epoch=0
        )
        
        self.assertFalse(should_stop)  # Should not stop on first epoch
        
        # Test no improvement scenario
        for epoch in range(1, 7):
            should_stop = early_stopping.check_early_stopping(
                self.model, 0.64, 0.54, action_recalls, severity_recalls, epoch=epoch
            )
        
        self.assertTrue(should_stop)  # Should stop after patience exceeded


class TestCurriculumLearningSystem(unittest.TestCase):
    """Test cases for Curriculum Learning System components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.total_epochs = 20
        
        # Create curriculum stages
        self.stages = [
            CurriculumStage(
                stage_name="easy",
                start_epoch=0,
                end_epoch=7,
                included_action_classes=[0, 1, 2, 3, 5, 6],
                included_severity_classes=[0, 1, 2],
                sampling_weights={'action': {}, 'severity': {}},
                loss_weights={'action': {}, 'severity': {}}
            ),
            CurriculumStage(
                stage_name="medium",
                start_epoch=8,
                end_epoch=14,
                included_action_classes=[0, 1, 2, 3, 4, 5, 6],
                included_severity_classes=[0, 1, 2, 3],
                sampling_weights={'action': {4: 2.0}, 'severity': {3: 2.0}},
                loss_weights={'action': {4: 1.5}, 'severity': {3: 1.5}}
            ),
            CurriculumStage(
                stage_name="hard",
                start_epoch=15,
                end_epoch=19,
                included_action_classes=list(range(8)),
                included_severity_classes=list(range(4)),
                sampling_weights={'action': {4: 3.0, 7: 3.0}, 'severity': {3: 3.0}},
                loss_weights={'action': {4: 2.0, 7: 2.0}, 'severity': {3: 2.0}}
            )
        ]
    
    def test_curriculum_scheduler(self):
        """Test CurriculumScheduler functionality"""
        
        scheduler = CurriculumScheduler(
            stages=self.stages,
            total_epochs=self.total_epochs
        )
        
        # Test initial state
        self.assertEqual(scheduler.total_epochs, self.total_epochs)
        self.assertEqual(len(scheduler.stages), 3)
        
        # Test stage progression
        for epoch in range(self.total_epochs):
            current_stage = scheduler.get_current_stage(epoch)
            stage_progress = scheduler.get_stage_progress(epoch)
            
            self.assertIsInstance(current_stage, CurriculumStage)
            self.assertIsInstance(stage_progress, float)
            self.assertGreaterEqual(stage_progress, 0.0)
            self.assertLessEqual(stage_progress, 1.0)
        
        # Test specific stage transitions
        self.assertEqual(scheduler.get_current_stage(0).stage_name, "easy")
        self.assertEqual(scheduler.get_current_stage(10).stage_name, "medium")
        self.assertEqual(scheduler.get_current_stage(18).stage_name, "hard")
        
        # Test curriculum stats
        stats = scheduler.get_curriculum_stats()
        self.assertIn('current_stage', stats)
        self.assertIn('stage_progress', stats)
        self.assertIn('total_stages', stats)
    
    def test_curriculum_learning_manager(self):
        """Test CurriculumLearningManager functionality"""
        
        manager = CurriculumLearningManager(
            total_epochs=self.total_epochs,
            stages=self.stages,
            save_dir=tempfile.mkdtemp()
        )
        
        # Test epoch configuration
        for epoch in [0, 10, 18]:
            config = manager.get_epoch_configuration(epoch)
            
            self.assertIn('current_stage', config)
            self.assertIn('included_classes', config)
            self.assertIn('sampling_weights', config)
            self.assertIn('loss_weights', config)
            self.assertIn('curriculum_stats', config)
        
        # Test performance metrics update
        metrics = {
            'combined_macro_recall': 0.45,
            'action_macro_recall': 0.5,
            'severity_macro_recall': 0.4,
            'action_class_recall': [0.6, 0.7, 0.8, 0.5, 0.1, 0.6, 0.7, 0.2],
            'severity_class_recall': [0.8, 0.6, 0.4, 0.1]
        }
        
        manager.update_performance_metrics(10, metrics)
        
        # Test checkpoint saving
        checkpoint_path = manager.save_checkpoint(10)
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Cleanup
        shutil.rmtree(manager.save_dir)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for Performance Monitor components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_class_names = {i: f"action_{i}" for i in range(8)}
        self.severity_class_names = {i: f"severity_{i}" for i in range(4)}
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        
        monitor = PerformanceMonitor(
            action_class_names=self.action_class_names,
            severity_class_names=self.severity_class_names,
            log_dir=self.temp_dir
        )
        
        # Test initialization
        self.assertEqual(monitor.action_class_names, self.action_class_names)
        self.assertEqual(monitor.severity_class_names, self.severity_class_names)
        self.assertEqual(len(monitor.epoch_metrics_history), 0)
        self.assertEqual(len(monitor.alerts_history), 0)
    
    def test_metrics_update(self):
        """Test metrics update functionality"""
        
        monitor = PerformanceMonitor(
            action_class_names=self.action_class_names,
            severity_class_names=self.severity_class_names,
            log_dir=self.temp_dir
        )
        
        # Create mock predictions and targets
        batch_size = 8
        action_predictions = torch.randn(batch_size, 8)
        action_targets = torch.randint(0, 8, (batch_size,))
        severity_predictions = torch.randn(batch_size, 4)
        severity_targets = torch.randint(0, 4, (batch_size,))
        
        # Update metrics
        epoch_metrics = monitor.update_metrics(
            epoch=0,
            action_predictions=action_predictions,
            action_targets=action_targets,
            severity_predictions=severity_predictions,
            severity_targets=severity_targets,
            loss_action=0.5,
            loss_severity=0.4,
            learning_rate=1e-4
        )
        
        # Check metrics
        self.assertIsInstance(epoch_metrics, EpochMetrics)
        self.assertEqual(epoch_metrics.epoch, 0)
        self.assertIsInstance(epoch_metrics.action_metrics, dict)
        self.assertIsInstance(epoch_metrics.severity_metrics, dict)
        
        # Check that metrics were stored
        self.assertEqual(len(monitor.epoch_metrics_history), 1)
    
    def test_alert_generation(self):
        """Test alert generation functionality"""
        
        monitor = PerformanceMonitor(
            action_class_names=self.action_class_names,
            severity_class_names=self.severity_class_names,
            log_dir=self.temp_dir,
            alert_thresholds={'zero_recall_epochs': 2}
        )
        
        # Create predictions that will result in zero recall for some classes
        batch_size = 8
        
        # Create biased predictions (always predict class 0)
        action_predictions = torch.zeros(batch_size, 8)
        action_predictions[:, 0] = 10.0  # High logit for class 0
        
        severity_predictions = torch.zeros(batch_size, 4)
        severity_predictions[:, 0] = 10.0  # High logit for class 0
        
        # Targets include other classes
        action_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        severity_targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        
        # Update metrics for multiple epochs to trigger alerts
        for epoch in range(3):
            monitor.update_metrics(
                epoch=epoch,
                action_predictions=action_predictions,
                action_targets=action_targets,
                severity_predictions=severity_predictions,
                severity_targets=severity_targets,
                loss_action=0.5,
                loss_severity=0.4,
                learning_rate=1e-4
            )
        
        # Check that alerts were generated
        self.assertGreater(len(monitor.alerts_history), 0)
        
        # Check alert types
        alert_types = [alert.alert_type.value for alert in monitor.alerts_history]
        self.assertIn('zero_recall', alert_types)


class TestModelEnsembleSystem(unittest.TestCase):
    """Test cases for Model Ensemble System components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_manager(self):
        """Test CheckpointManager functionality"""
        
        from model_ensemble_system import CheckpointManager, EnsembleConfig
        
        config = EnsembleConfig(
            max_checkpoints=3,
            min_performance_threshold=0.3,
            save_dir=self.temp_dir
        )
        
        manager = CheckpointManager(config)
        
        # Test adding checkpoints
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        for i in range(5):
            metrics = {'combined_macro_recall': 0.3 + i * 0.1}
            checkpoint_path = manager.add_checkpoint(
                model, optimizer, i, metrics
            )
            
            if metrics['combined_macro_recall'] >= config.min_performance_threshold:
                self.assertIsNotNone(checkpoint_path)
                self.assertTrue(os.path.exists(checkpoint_path))
        
        # Check that only max_checkpoints are kept
        self.assertLessEqual(len(manager.checkpoints), config.max_checkpoints)
        
        # Test checkpoint retrieval
        best_checkpoint = manager.get_best_checkpoints(2)
        self.assertLessEqual(len(best_checkpoint), 2)
    
    def test_ensemble_predictor(self):
        """Test EnsemblePredictor functionality"""
        
        from model_ensemble_system import EnsemblePredictor
        
        # Create mock models
        models = [nn.Linear(10, 2) for _ in range(3)]
        model_weights = [0.4, 0.3, 0.3]
        
        predictor = EnsemblePredictor(models, model_weights, self.device)
        
        # Test prediction
        input_data = torch.randn(4, 10)
        predictions = predictor.predict(input_data)
        
        self.assertEqual(predictions.shape, (4, 2))
        
        # Test detailed prediction
        detailed_predictions = predictor.predict_with_details(input_data)
        
        self.assertIn('ensemble_prediction', detailed_predictions)
        self.assertIn('individual_predictions', detailed_predictions)
        self.assertIn('prediction_variance', detailed_predictions)
        self.assertIn('model_weights', detailed_predictions)


class TestHyperparameterOptimizer(unittest.TestCase):
    """Test cases for Hyperparameter Optimizer components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hyperparameter_space(self):
        """Test HyperparameterSpace functionality"""
        
        # Test continuous parameter
        continuous_param = HyperparameterSpace(
            name="learning_rate",
            param_type=HyperparameterType.CONTINUOUS,
            low=1e-5,
            high=1e-2,
            default=1e-4
        )
        
        self.assertEqual(continuous_param.name, "learning_rate")
        self.assertEqual(continuous_param.param_type, HyperparameterType.CONTINUOUS)
        self.assertEqual(continuous_param.low, 1e-5)
        self.assertEqual(continuous_param.high, 1e-2)
        self.assertEqual(continuous_param.default, 1e-4)
        
        # Test categorical parameter
        categorical_param = HyperparameterSpace(
            name="optimizer",
            param_type=HyperparameterType.CATEGORICAL,
            categories=["adam", "adamw", "sgd"],
            default="adamw"
        )
        
        self.assertEqual(categorical_param.categories, ["adam", "adamw", "sgd"])
        self.assertEqual(categorical_param.default, "adamw")
    
    def test_automated_hyperparameter_optimizer(self):
        """Test AutomatedHyperparameterOptimizer functionality"""
        
        from hyperparameter_optimizer import OptimizationStrategy
        
        # Create parameter space
        parameter_space = [
            HyperparameterSpace(
                name="learning_rate",
                param_type=HyperparameterType.CONTINUOUS,
                low=1e-5,
                high=1e-2,
                default=1e-4
            ),
            HyperparameterSpace(
                name="batch_size",
                param_type=HyperparameterType.INTEGER,
                low=2,
                high=8,
                default=4
            )
        ]
        
        optimizer = AutomatedHyperparameterOptimizer(
            parameter_space=parameter_space,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            save_dir=self.temp_dir
        )
        
        # Test objective function
        def mock_objective(params):
            # Simple mock objective that prefers higher learning rates
            return params.get('learning_rate', 1e-4) * 1000
        
        # Test optimization (with few calls for speed)
        result = optimizer.optimize_hyperparameters(
            objective_function=mock_objective,
            n_calls=3
        )
        
        self.assertIsNotNone(result.best_params)
        self.assertIsNotNone(result.best_score)
        self.assertIn('learning_rate', result.best_params)
        self.assertIn('batch_size', result.best_params)


class TestIntegrationModules(unittest.TestCase):
    """Test cases for Integration Modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_adaptive_loss_wrapper(self):
        """Test AdaptiveLossWrapper integration"""
        
        wrapper = AdaptiveLossWrapper(
            num_action_classes=8,
            num_severity_classes=4,
            use_adaptive_system=True
        )
        
        # Test loss computation
        batch_size = 4
        action_logits = torch.randn(batch_size, 8, requires_grad=True)
        severity_logits = torch.randn(batch_size, 4, requires_grad=True)
        action_targets = torch.randint(0, 8, (batch_size,))
        severity_targets = torch.randint(0, 4, (batch_size,))
        
        total_loss, loss_components = wrapper.compute_loss(
            action_logits, severity_logits, action_targets, severity_targets
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_components, dict)
        
        # Test backward pass
        total_loss.backward()
        self.assertIsNotNone(action_logits.grad)
        self.assertIsNotNone(severity_logits.grad)
        
        # Test criterion functions
        action_criterion, severity_criterion = wrapper.get_criterion_functions()
        self.assertIsInstance(action_criterion, nn.Module)
        self.assertIsInstance(severity_criterion, nn.Module)
    
    @patch('complete_training_integration.create_enhanced_training_setup')
    @patch('complete_training_integration.MVFoulsModel')
    def test_complete_enhanced_training_system(self, mock_model, mock_setup):
        """Test CompleteEnhancedTrainingSystem integration"""
        
        # Mock the enhanced training setup
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_stats = {'synthetic_samples_generated': 50}
        mock_setup.return_value = (mock_train_loader, mock_val_loader, mock_stats)
        
        # Mock the model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        # Create test configuration
        config = {
            'data': {
                'folder': 'test_data',
                'train_split': 'train',
                'val_split': 'test',
                'start_frame': 0,
                'end_frame': 10
            },
            'model': {
                'use_enhanced': False,
                'aggregation': 'attention',
                'input_size': [224, 224]
            },
            'training': {
                'epochs': 2,
                'batch_size': 4,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'num_workers': 0,
                'accumulation_steps': 2,
                'early_stopping_patience': 5
            },
            'enhanced_pipeline': {},
            'adaptive_loss': {},
            'curriculum_learning': {'enabled': False},
            'performance_monitoring': {},
            'ensemble_system': {'enabled': False},
            'hyperparameter_optimization': {'enabled': False}
        }
        
        # Test system initialization
        system = CompleteEnhancedTrainingSystem(config, self.temp_dir)
        
        self.assertEqual(system.config, config)
        self.assertEqual(system.save_dir, self.temp_dir)
        
        # Test component setup
        train_loader, val_loader, stats = system.setup_enhanced_data_pipeline()
        self.assertEqual(train_loader, mock_train_loader)
        self.assertEqual(val_loader, mock_val_loader)
        self.assertEqual(stats, mock_stats)
        
        # Test adaptive loss setup
        adaptive_loss = system.setup_adaptive_loss_system()
        self.assertIsInstance(adaptive_loss, AdaptiveLossWrapper)
        
        # Test performance monitoring setup
        monitor = system.setup_performance_monitoring()
        self.assertIsInstance(monitor, PerformanceMonitor)


def run_component_tests():
    """Run all component tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEnhancedDataPipeline,
        TestAdaptiveLossSystem,
        TestAdvancedTrainingStrategies,
        TestCurriculumLearningSystem,
        TestPerformanceMonitor,
        TestModelEnsembleSystem,
        TestHyperparameterOptimizer,
        TestIntegrationModules
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("Running comprehensive unit tests for enhanced training components...")
    
    # Run all tests
    result = run_component_tests()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n✓ All tests passed successfully!")
    else:
        print(f"\n✗ Some tests failed. Please review the output above.")