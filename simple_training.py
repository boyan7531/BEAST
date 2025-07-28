#!/usr/bin/env python3
"""
Simple Enhanced Training Script

This script provides a straightforward way to run the enhanced training
with all the fixes applied.
"""

import logging
import torch
from complete_training_integration import CompleteEnhancedTrainingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_working_config(epochs=5, batch_size=4):
    """Create a working configuration"""
    return {
        'data': {
            'folder': 'mvfouls',
            'train_split': 'train',
            'val_split': 'test',
            'start_frame': 63,
            'end_frame': 86
        },
        'model': {
            'use_enhanced': False,  # Use standard model for reliability
            'aggregation': 'attention',
            'input_size': [224, 224]
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_workers': 8,  # Use multiple workers for faster data loading
            'accumulation_steps': 4,
            'early_stopping_patience': 10
        },
        'enhanced_pipeline': {
            'enabled': True,
            'synthetic_generation': {
                'target_classes': {
                    'action': {4: 20, 7: 20},  # Pushing: 20, Dive: 20
                    'severity': {3: 40}        # Red Card: 40
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
            'enabled': False  # Keep disabled for stability
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
            'enabled': False  # Keep disabled for stability
        },
        'hyperparameter_optimization': {
            'enabled': False  # Keep disabled for stability
        },
        'save_dir': 'enhanced_training_results',
        'experiment_name': 'mvfouls_enhanced_training',
        'random_seed': 42
    }


def run_enhanced_training(epochs=10, batch_size=4):
    """Run enhanced training with the working configuration"""
    
    print("🚀 Starting Enhanced MVFouls Training")
    print("=" * 50)
    
    # Create configuration
    config = create_working_config(epochs, batch_size)
    
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Enhanced Pipeline: ✅ Enabled")
    print(f"  Adaptive Loss: ✅ Enabled")
    print(f"  Performance Monitoring: ✅ Enabled")
    print(f"  Save Directory: {config['save_dir']}")
    print()
    
    try:
        # Create training system
        system = CompleteEnhancedTrainingSystem(config, config['save_dir'])
        
        # Setup components
        print("Setting up enhanced components...")
        
        # 1. Enhanced Data Pipeline
        train_dataloader, val_dataloader, pipeline_stats = system.setup_enhanced_data_pipeline()
        print(f"✅ Data Pipeline: {len(train_dataloader.dataset)} train, {len(val_dataloader.dataset)} val, {pipeline_stats.get('synthetic_samples_generated', 0)} synthetic")
        
        # 2. Adaptive Loss System
        adaptive_loss = system.setup_adaptive_loss_system()
        print("✅ Adaptive Loss System")
        
        # 3. Model and Optimizer
        model, optimizer = system.setup_model_and_optimizer()
        print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # 4. Performance Monitoring
        performance_monitor = system.setup_performance_monitoring()
        print("✅ Performance Monitoring")
        
        print("\n🎯 Starting Training Loop...")
        print("=" * 50)
        
        # Training loop
        best_combined_recall = 0.0
        device = system.device
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (videos, action_labels, severity_labels, action_ids) in enumerate(train_dataloader):
                # Move data to device
                videos = [video.to(device) for video in videos]
                action_labels = action_labels.to(device)
                severity_labels = severity_labels.to(device)
                
                # Forward pass
                action_logits, severity_logits = model(videos)
                
                # Compute loss
                total_loss, loss_components = adaptive_loss.compute_loss(
                    action_logits, severity_logits,
                    torch.argmax(action_labels, dim=1),
                    torch.argmax(severity_labels, dim=1)
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                # Log progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_dataloader)}: Loss = {total_loss.item():.4f}")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            all_action_preds = []
            all_action_targets = []
            all_severity_preds = []
            all_severity_targets = []
            
            with torch.no_grad():
                for videos, action_labels, severity_labels, action_ids in val_dataloader:
                    videos = [video.to(device) for video in videos]
                    action_labels = action_labels.to(device)
                    severity_labels = severity_labels.to(device)
                    
                    action_logits, severity_logits = model(videos)
                    
                    total_loss, _ = adaptive_loss.compute_loss(
                        action_logits, severity_logits,
                        torch.argmax(action_labels, dim=1),
                        torch.argmax(severity_labels, dim=1)
                    )
                    
                    val_loss += total_loss.item()
                    val_batches += 1
                    
                    # Collect predictions for metrics
                    all_action_preds.append(action_logits.cpu())
                    all_action_targets.append(action_labels.cpu())
                    all_severity_preds.append(severity_logits.cpu())
                    all_severity_targets.append(severity_labels.cpu())
            
            avg_val_loss = val_loss / val_batches
            
            # Calculate metrics
            action_preds = torch.cat(all_action_preds, dim=0)
            action_targets = torch.cat(all_action_targets, dim=0)
            severity_preds = torch.cat(all_severity_preds, dim=0)
            severity_targets = torch.cat(all_severity_targets, dim=0)
            
            # Convert to class predictions
            action_pred_classes = torch.argmax(action_preds, dim=1)
            action_true_classes = torch.argmax(action_targets, dim=1)
            severity_pred_classes = torch.argmax(severity_preds, dim=1)
            severity_true_classes = torch.argmax(severity_targets, dim=1)
            
            # Calculate per-class recalls
            from sklearn.metrics import precision_recall_fscore_support
            
            action_precision, action_recall, _, _ = precision_recall_fscore_support(
                action_true_classes.numpy(), action_pred_classes.numpy(), 
                labels=list(range(8)), average=None, zero_division=0
            )
            
            severity_precision, severity_recall, _, _ = precision_recall_fscore_support(
                severity_true_classes.numpy(), severity_pred_classes.numpy(),
                labels=list(range(4)), average=None, zero_division=0
            )
            
            # Calculate macro recalls
            action_macro_recall = action_recall.mean()
            severity_macro_recall = severity_recall.mean()
            combined_macro_recall = (action_macro_recall + severity_macro_recall) / 2
            
            # Update best performance
            if combined_macro_recall > best_combined_recall:
                best_combined_recall = combined_macro_recall
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'combined_macro_recall': combined_macro_recall,
                    'action_recalls': action_recall.tolist(),
                    'severity_recalls': severity_recall.tolist()
                }, f"{config['save_dir']}/best_model.pth")
            
            # Print epoch results
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Action Macro Recall: {action_macro_recall:.4f}")
            print(f"Severity Macro Recall: {severity_macro_recall:.4f}")
            print(f"Combined Macro Recall: {combined_macro_recall:.4f} (Best: {best_combined_recall:.4f})")
            
            # Print minority class performance
            print(f"Minority Class Performance:")
            print(f"  Pushing (Action 4): {action_recall[4]:.4f}")
            print(f"  Dive (Action 7): {action_recall[7]:.4f}")
            print(f"  Red Card (Severity 3): {severity_recall[3]:.4f}")
        
        print("\n🎉 Training Completed!")
        print("=" * 50)
        print(f"Best Combined Macro Recall: {best_combined_recall:.4f}")
        print(f"Target: 0.45 ({'✅ ACHIEVED' if best_combined_recall >= 0.45 else '❌ NOT ACHIEVED'})")
        print(f"Results saved to: {config['save_dir']}")
        
        return best_combined_recall
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Enhanced Training")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    # Run training
    result = run_enhanced_training(args.epochs, args.batch_size)
    
    if result is not None:
        print(f"\n✅ Training completed successfully!")
        print(f"Final performance: {result:.4f}")
    else:
        print(f"\n❌ Training failed!")
        exit(1)