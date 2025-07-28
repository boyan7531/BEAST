#!/usr/bin/env python3
"""
Enhanced Training Runner for MVFouls Performance Optimization

This script provides an easy way to run the complete enhanced training pipeline
with different configuration options.

Usage:
    python run_enhanced_training.py --mode quick_test
    python run_enhanced_training.py --mode production
    python run_enhanced_training.py --config custom_config.json
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Import the enhanced training system
from complete_training_integration import (
    CompleteEnhancedTrainingSystem,
    run_complete_enhanced_training
)
from enhanced_training_config import (
    EnhancedTrainingConfig,
    ConfigurationManager,
    create_default_config,
    create_quick_test_config,
    create_production_config
)
from enhanced_training_error_handling import (
    ErrorHandler,
    error_handling_context
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met for training"""
    logger.info("Checking prerequisites...")
    
    # Check if dataset exists
    if not os.path.exists("mvfouls"):
        logger.error("❌ Dataset folder 'mvfouls' not found!")
        logger.error("Please ensure the MVFouls dataset is available in the current directory")
        return False
    
    # Check for required splits
    required_splits = ["train", "test"]
    for split in required_splits:
        split_path = os.path.join("mvfouls", split)
        if not os.path.exists(split_path):
            logger.error(f"❌ Dataset split '{split}' not found at {split_path}")
            return False
    
    # Check PyTorch availability
    try:
        import torch
        logger.info(f"✅ PyTorch available: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ️  CUDA not available, will use CPU")
    except ImportError:
        logger.error("❌ PyTorch not installed!")
        return False
    
    # Check other required packages
    required_packages = ['numpy', 'sklearn', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {missing_packages}")
        logger.error("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All prerequisites met!")
    return True


def run_quick_test():
    """Run a quick test training session"""
    logger.info("🚀 Starting quick test training...")
    
    # Create quick test configuration
    config = create_quick_test_config()
    
    # Override some settings for even faster testing
    config.training.epochs = 2
    config.training.batch_size = 2
    config.enhanced_pipeline.synthetic_generation['target_classes'] = {
        'action': {4: 2, 7: 2},
        'severity': {3: 5}
    }
    
    logger.info("Configuration:")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch Size: {config.training.batch_size}")
    logger.info(f"  Save Directory: {config.save_dir}")
    
    # Run training
    try:
        results = run_complete_enhanced_training(
            config=config.to_dict(),
            save_dir=config.save_dir
        )
        
        logger.info("🎉 Quick test training completed successfully!")
        logger.info(f"Results saved to: {config.save_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Quick test training failed: {e}")
        raise


def run_production_training():
    """Run full production training"""
    logger.info("🚀 Starting production training...")
    
    # Create production configuration
    config = create_production_config()
    
    # Validate configuration
    manager = ConfigurationManager(config)
    if not manager.validate_configuration():
        logger.error("❌ Configuration validation failed!")
        for error in manager.validation_errors:
            logger.error(f"  - {error}")
        return None
    
    # Setup environment
    manager.setup_environment()
    
    logger.info("Production Configuration:")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch Size: {config.training.batch_size}")
    logger.info(f"  Learning Rate: {config.training.learning_rate}")
    logger.info(f"  Save Directory: {config.save_dir}")
    logger.info(f"  Components Enabled:")
    logger.info(f"    - Enhanced Pipeline: {config.enhanced_pipeline.enabled}")
    logger.info(f"    - Adaptive Loss: {config.adaptive_loss.enabled}")
    logger.info(f"    - Curriculum Learning: {config.curriculum_learning.enabled}")
    logger.info(f"    - Performance Monitoring: {config.performance_monitoring.enabled}")
    logger.info(f"    - Ensemble System: {config.ensemble_system.enabled}")
    
    # Run training
    try:
        results = run_complete_enhanced_training(
            config=config.to_dict(),
            save_dir=config.save_dir
        )
        
        logger.info("🎉 Production training completed successfully!")
        logger.info(f"Results saved to: {config.save_dir}")
        
        # Print final results
        if 'integration_metrics' in results:
            best_performance = results['integration_metrics'].get('best_performance', 0.0)
            target_achieved = results['integration_metrics'].get('target_achievement', {}).get('all_targets_achieved', False)
            
            logger.info(f"📊 Final Results:")
            logger.info(f"  Best Combined Macro Recall: {best_performance:.4f}")
            logger.info(f"  Target Achievement: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Production training failed: {e}")
        raise


def run_custom_config(config_path: str):
    """Run training with custom configuration file"""
    logger.info(f"🚀 Starting training with custom config: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"❌ Configuration file not found: {config_path}")
        return None
    
    try:
        # Load configuration
        config = EnhancedTrainingConfig.load(config_path)
        
        # Validate configuration
        manager = ConfigurationManager(config)
        if not manager.validate_configuration():
            logger.error("❌ Configuration validation failed!")
            for error in manager.validation_errors:
                logger.error(f"  - {error}")
            return None
        
        # Setup environment
        manager.setup_environment()
        
        logger.info("Custom Configuration Loaded:")
        logger.info(f"  Experiment: {config.experiment_name}")
        logger.info(f"  Epochs: {config.training.epochs}")
        logger.info(f"  Batch Size: {config.training.batch_size}")
        
        # Run training
        results = run_complete_enhanced_training(
            config=config.to_dict(),
            save_dir=config.save_dir
        )
        
        logger.info("🎉 Custom training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Custom training failed: {e}")
        raise


def create_sample_config():
    """Create a sample configuration file for customization"""
    logger.info("📝 Creating sample configuration file...")
    
    config = create_default_config()
    config.experiment_name = "my_custom_experiment"
    config.save_dir = "my_training_results"
    
    # Save sample config
    config.save("sample_config.json")
    
    logger.info("✅ Sample configuration created: sample_config.json")
    logger.info("You can modify this file and run with: --config sample_config.json")


def main():
    """Main training runner"""
    parser = argparse.ArgumentParser(
        description="Enhanced Training Runner for MVFouls Performance Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_enhanced_training.py --mode quick_test     # Quick test (2 epochs)
  python run_enhanced_training.py --mode production    # Full production training
  python run_enhanced_training.py --config my_config.json  # Custom configuration
  python run_enhanced_training.py --create-sample      # Create sample config file
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick_test', 'production'],
        help='Training mode to use'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample configuration file'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip prerequisite checks (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Handle sample config creation
    if args.create_sample:
        create_sample_config()
        return
    
    # Check if any training option is specified
    if not args.mode and not args.config:
        parser.print_help()
        logger.error("❌ Please specify either --mode or --config")
        sys.exit(1)
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            logger.error("❌ Prerequisites not met. Use --skip-checks to bypass (not recommended)")
            sys.exit(1)
    
    # Create error handler
    error_handler = ErrorHandler(log_dir="training_error_logs")
    
    # Run training with error handling
    with error_handling_context(error_handler, reraise=True):
        try:
            if args.mode == 'quick_test':
                results = run_quick_test()
            elif args.mode == 'production':
                results = run_production_training()
            elif args.config:
                results = run_custom_config(args.config)
            
            if results:
                logger.info("🎉 Training completed successfully!")
                logger.info("📊 Check the results directory for detailed outputs")
            else:
                logger.error("❌ Training failed!")
                sys.exit(1)
                
        except KeyboardInterrupt:
            logger.info("⏹️  Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            logger.error("📋 Check training_error_logs/ for detailed error information")
            sys.exit(1)


if __name__ == "__main__":
    main()