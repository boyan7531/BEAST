"""
Simple Integration Test for Enhanced Training Components

This module provides a simplified test to verify that all components
can be imported and basic functionality works.
"""

import sys
import traceback
import torch
import numpy as np
from typing import Dict, Any

def test_imports():
    """Test that all components can be imported"""
    print("Testing component imports...")
    
    try:
        # Test enhanced data pipeline
        from enhanced_data_pipeline import StratifiedMinorityBatchSampler, MixupVideoGenerator
        print("✓ Enhanced data pipeline imports successful")
    except Exception as e:
        print(f"✗ Enhanced data pipeline import failed: {e}")
        return False
    
    try:
        # Test adaptive loss system
        from adaptive_loss_system import DynamicLossSystem, AdaptiveFocalLoss
        print("✓ Adaptive loss system imports successful")
    except Exception as e:
        print(f"✗ Adaptive loss system import failed: {e}")
        return False
    
    try:
        # Test advanced training strategies
        from advanced_training_strategies import AdvancedTrainingStrategiesManager
        print("✓ Advanced training strategies imports successful")
    except Exception as e:
        print(f"✗ Advanced training strategies import failed: {e}")
        return False
    
    try:
        # Test curriculum learning
        from curriculum_learning_system import CurriculumLearningManager
        print("✓ Curriculum learning imports successful")
    except Exception as e:
        print(f"✗ Curriculum learning import failed: {e}")
        return False
    
    try:
        # Test performance monitor
        from performance_monitor import PerformanceMonitor
        print("✓ Performance monitor imports successful")
    except Exception as e:
        print(f"✗ Performance monitor import failed: {e}")
        return False
    
    try:
        # Test ensemble system
        from model_ensemble_system import EnsembleSystem
        print("✓ Ensemble system imports successful")
    except Exception as e:
        print(f"✗ Ensemble system import failed: {e}")
        return False
    
    try:
        # Test hyperparameter optimizer
        from hyperparameter_optimizer import AutomatedHyperparameterOptimizer
        print("✓ Hyperparameter optimizer imports successful")
    except Exception as e:
        print(f"✗ Hyperparameter optimizer import failed: {e}")
        return False
    
    try:
        # Test integration modules
        from enhanced_pipeline_integration import EnhancedDataPipelineManager
        from adaptive_loss_integration import AdaptiveLossWrapper
        print("✓ Integration modules imports successful")
    except Exception as e:
        print(f"✗ Integration modules import failed: {e}")
        return False
    
    try:
        # Test configuration and error handling
        from enhanced_training_config import EnhancedTrainingConfig
        from enhanced_training_error_handling import ErrorHandler
        print("✓ Configuration and error handling imports successful")
    except Exception as e:
        print(f"✗ Configuration and error handling import failed: {e}")
        return False
    
    try:
        # Test complete integration
        from complete_training_integration import CompleteEnhancedTrainingSystem
        print("✓ Complete integration imports successful")
    except Exception as e:
        print(f"✗ Complete integration import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test adaptive loss system
        from adaptive_loss_system import DynamicLossSystem
        
        loss_system = DynamicLossSystem(
            num_action_classes=8,
            num_severity_classes=4
        )
        
        # Test forward pass
        batch_size = 4
        action_logits = torch.randn(batch_size, 8, requires_grad=True)
        severity_logits = torch.randn(batch_size, 4, requires_grad=True)
        action_targets = torch.randint(0, 8, (batch_size,))
        severity_targets = torch.randint(0, 4, (batch_size,))
        
        total_loss, loss_components = loss_system(
            action_logits, severity_logits, action_targets, severity_targets
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        print("✓ Adaptive loss system basic functionality works")
        
    except Exception as e:
        print(f"✗ Adaptive loss system test failed: {e}")
        return False
    
    try:
        # Test performance monitor
        from performance_monitor import PerformanceMonitor
        
        action_class_names = {i: f"action_{i}" for i in range(8)}
        severity_class_names = {i: f"severity_{i}" for i in range(4)}
        
        monitor = PerformanceMonitor(
            action_class_names=action_class_names,
            severity_class_names=severity_class_names,
            log_dir="test_logs"
        )
        
        # Test metrics update
        batch_size = 8
        action_predictions = torch.randn(batch_size, 8)
        action_targets = torch.randint(0, 8, (batch_size,))
        severity_predictions = torch.randn(batch_size, 4)
        severity_targets = torch.randint(0, 4, (batch_size,))
        
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
        
        assert epoch_metrics is not None
        print("✓ Performance monitor basic functionality works")
        
    except Exception as e:
        print(f"✗ Performance monitor test failed: {e}")
        return False
    
    try:
        # Test configuration system
        from enhanced_training_config import EnhancedTrainingConfig, ConfigurationManager
        
        config = EnhancedTrainingConfig()
        manager = ConfigurationManager(config)
        
        # Test validation
        is_valid = manager.validate_configuration()
        assert isinstance(is_valid, bool)
        
        # Test configuration summary
        summary = manager.get_config_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        print("✓ Configuration system basic functionality works")
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False
    
    try:
        # Test error handling system
        from enhanced_training_error_handling import ErrorHandler, TrainingError, ErrorCategory
        
        error_handler = ErrorHandler(log_dir="test_error_logs")
        
        # Test error handling
        test_error = TrainingError(
            message="Test error",
            category=ErrorCategory.MODEL_TRAINING
        )
        
        handled = error_handler.handle_error(test_error, attempt_recovery=False)
        assert isinstance(handled, bool)
        
        # Test error summary
        summary = error_handler.get_error_summary()
        assert isinstance(summary, dict)
        assert 'total_errors' in summary
        
        print("✓ Error handling system basic functionality works")
        
    except Exception as e:
        print(f"✗ Error handling system test failed: {e}")
        return False
    
    return True


def test_integration_compatibility():
    """Test that integration modules work together"""
    print("\nTesting integration compatibility...")
    
    try:
        # Test adaptive loss wrapper
        from adaptive_loss_integration import AdaptiveLossWrapper
        
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
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        
        print("✓ Adaptive loss integration works")
        
    except Exception as e:
        print(f"✗ Adaptive loss integration test failed: {e}")
        return False
    
    try:
        # Test configuration with complete system
        from complete_training_integration import create_default_config
        from enhanced_training_config import ConfigurationManager
        
        config = create_default_config()
        manager = ConfigurationManager(config)
        
        # Test validation
        is_valid = manager.validate_configuration()
        assert isinstance(is_valid, bool)
        
        print("✓ Complete system configuration integration works")
        
    except Exception as e:
        print(f"✗ Complete system configuration test failed: {e}")
        return False
    
    return True


def cleanup_test_files():
    """Clean up test files created during testing"""
    import os
    import shutil
    
    test_dirs = ["test_logs", "test_error_logs"]
    test_files = ["test_config.json", "test_error_report.json"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except:
                pass
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except:
                pass


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("ENHANCED TRAINING COMPONENTS INTEGRATION TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test integration compatibility
    if not test_integration_compatibility():
        all_tests_passed = False
    
    # Cleanup
    cleanup_test_files()
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if all_tests_passed:
        print("✓ ALL TESTS PASSED!")
        print("✓ Enhanced training components are ready for integration")
        print("✓ All imports work correctly")
        print("✓ Basic functionality is operational")
        print("✓ Integration compatibility verified")
    else:
        print("✗ SOME TESTS FAILED!")
        print("✗ Please review the output above for specific failures")
        print("✗ Fix the issues before proceeding with full integration")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)