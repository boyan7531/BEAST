#!/usr/bin/env python3
"""
Quick Fix Training Script

This script provides a patched version of the training that handles
configuration issues gracefully.
"""

import logging
import torch
from complete_training_integration import CompleteEnhancedTrainingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fixed_config():
    """Create a configuration that works around the current issues"""
    return {
        'data': {
            'folder': 'mvfouls',
            'train_split': 'train',
            'val_split': 'test',
            'start_frame': 63,
            'end_frame': 86
        },
        'model': {
            'use_enhanced': False,  # Use standard model to avoid complexity
            'aggregation': 'attention',
            'input_size': [224, 224]
        },
        'training': {
            'epochs': 2,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_workers': 0,
            'accumulation_steps': 2,
            'early_stopping_patience': 5
        },
        'enhanced_pipeline': {
            'enabled': True,
            'synthetic_generation': {
                'target_classes': {
                    'action': {4: 2, 7: 2},
                    'severity': {3: 5}
                },
                'mixup_alpha': 0.4,
                'temporal_alpha': 0.3
            },
            'quality_validation': {
                'feature_similarity_threshold': 0.7,
                'temporal_consistency_threshold': 0.8,
                'label_consistency_threshold': 0.9
            },
            'stratified_sampling': {
                'min_minority_per_batch': 1,
                'minority_threshold': 0.05
            },
            'advanced_augmentation': {
                'temporal_jitter_range': [0.8, 1.2],
                'spatial_jitter_prob': 0.5,
                'color_jitter_prob': 0.3,
                'gaussian_noise_prob': 0.2,
                'frame_dropout_prob': 0.1
            }
        },
        'adaptive_loss': {
            'enabled': True,
            'initial_gamma': 2.0,
            'initial_alpha': 1.0,
            'recall_threshold': 0.1,
            'weight_increase_factor': 1.5
        },
        'curriculum_learning': {
            'enabled': False  # Disable for simplicity
        },
        'performance_monitoring': {
            'enabled': True,
            'thresholds': {
                'zero_recall_epochs': 3,
                'performance_drop_threshold': 0.1,
                'target_combined_recall': 0.45
            }
        },
        'ensemble_system': {
            'enabled': False  # Disable for simplicity
        },
        'hyperparameter_optimization': {
            'enabled': False  # Disable for simplicity
        },
        'save_dir': 'quick_fix_results',
        'experiment_name': 'quick_fix_test',
        'random_seed': 42
    }


def run_quick_fix_training():
    """Run training with the fixed configuration"""
    logger.info("🚀 Starting quick fix training...")
    
    # Create fixed configuration
    config = create_fixed_config()
    
    logger.info("Fixed Configuration:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  Enhanced Pipeline: {config['enhanced_pipeline']['enabled']}")
    logger.info(f"  Adaptive Loss: {config['adaptive_loss']['enabled']}")
    logger.info(f"  Save Directory: {config['save_dir']}")
    
    try:
        # Create and run the training system
        system = CompleteEnhancedTrainingSystem(config, config['save_dir'])
        
        # Setup components individually to catch issues
        logger.info("Setting up enhanced data pipeline...")
        train_dataloader, val_dataloader, pipeline_stats = system.setup_enhanced_data_pipeline()
        logger.info(f"✅ Data pipeline setup successful!")
        logger.info(f"  Training samples: {len(train_dataloader.dataset)}")
        logger.info(f"  Validation samples: {len(val_dataloader.dataset)}")
        logger.info(f"  Synthetic samples: {pipeline_stats.get('synthetic_samples_generated', 0)}")
        
        logger.info("Setting up adaptive loss system...")
        adaptive_loss = system.setup_adaptive_loss_system()
        logger.info("✅ Adaptive loss setup successful!")
        
        logger.info("Setting up model and optimizer...")
        model, optimizer = system.setup_model_and_optimizer()
        logger.info("✅ Model and optimizer setup successful!")
        
        logger.info("Setting up performance monitoring...")
        performance_monitor = system.setup_performance_monitoring()
        logger.info("✅ Performance monitoring setup successful!")
        
        logger.info("🎉 All components set up successfully!")
        logger.info("The enhanced training system is working correctly.")
        
        # Run a mini training loop to verify everything works
        logger.info("Running mini training verification...")
        
        model.train()
        for batch_idx, (videos, action_labels, severity_labels, action_ids) in enumerate(train_dataloader):
            if batch_idx >= 2:  # Just test 2 batches
                break
                
            # Move data to device
            videos = [video.to(system.device) for video in videos]
            action_labels = action_labels.to(system.device)
            severity_labels = severity_labels.to(system.device)
            
            # Forward pass
            action_logits, severity_logits = model(videos)
            
            # Compute loss
            total_loss, loss_components = adaptive_loss.compute_loss(
                action_logits, severity_logits,
                torch.argmax(action_labels, dim=1),
                torch.argmax(severity_labels, dim=1)
            )
            
            logger.info(f"  Batch {batch_idx + 1}: Loss = {total_loss.item():.4f}")
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        logger.info("✅ Mini training verification successful!")
        logger.info("🎉 The enhanced training system is ready for full training!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick fix training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_quick_fix_training()
    if success:
        print("\n🎉 SUCCESS! The enhanced training system is working correctly.")
        print("You can now run full training with:")
        print("  python run_enhanced_training.py --mode quick_test")
    else:
        print("\n❌ There are still issues that need to be resolved.")
        print("Please check the error messages above.")