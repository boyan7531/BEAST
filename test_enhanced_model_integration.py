#!/usr/bin/env python3
"""
Test script to verify the enhanced model works with real MVFouls data.
"""

import torch
import os
from enhanced_model import EnhancedMVFoulsModel
from dataset import MVFoulsDataset, custom_collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from transform import get_val_transforms


def test_enhanced_model_with_real_data():
    print("Testing Enhanced MVFouls Model with real data...")
    
    # Check if dataset exists
    test_folder = "mvfouls"
    if not os.path.exists(test_folder):
        print(f"Dataset folder '{test_folder}' not found. Skipping real data test.")
        return
    
    # Initialize model
    model = EnhancedMVFoulsModel()
    print("✓ Enhanced model instantiated successfully")
    
    # Create dataset and dataloader
    try:
        transform = get_val_transforms((224, 224))
        dataset = MVFoulsDataset(
            test_folder, 
            "train", 
            start_frame=63, 
            end_frame=86, 
            transform_model=transform, 
            target_fps=17
        )
        
        if len(dataset) == 0:
            print("No samples found in dataset. Skipping real data test.")
            return
            
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=custom_collate_fn
        )
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Test with real batch
    try:
        batch_videos, batch_action_labels, batch_severity_labels, _ = next(iter(dataloader))
        print(f"✓ Got real batch with {len(batch_videos)} samples")
        
        # Forward pass
        action_logits, severity_logits, confidence_scores, attention_info = model(
            batch_videos, return_attention=True
        )
        
        print("✓ Forward pass with real data successful")
        print(f"  Action logits shape: {action_logits.shape}")
        print(f"  Severity logits shape: {severity_logits.shape}")
        print(f"  Expected action labels shape: {batch_action_labels.shape}")
        print(f"  Expected severity labels shape: {batch_severity_labels.shape}")
        
        # Verify shapes match expected outputs
        batch_size = len(batch_videos)
        assert action_logits.shape == (batch_size, 8), f"Action logits shape mismatch: {action_logits.shape}"
        assert severity_logits.shape == (batch_size, 4), f"Severity logits shape mismatch: {severity_logits.shape}"
        
        # Test confidence scores
        for key, value in confidence_scores.items():
            assert value.shape[0] == batch_size, f"Confidence score {key} batch size mismatch: {value.shape}"
            print(f"  {key}: {value.shape}")
        
        # Test attention info
        print(f"  Clip importance weights: {len(attention_info['clip_importance_weights'])} samples")
        print(f"  Diversity loss: {attention_info['diversity_loss'].item():.4f}")
        
        # Test class attention weights
        class_weights = model.get_class_attention_weights()
        print(f"  Action class queries shape: {class_weights['action_class_queries'].shape}")
        print(f"  Severity class queries shape: {class_weights['severity_class_queries'].shape}")
        
        print("✓ All integration tests passed!")
        
    except Exception as e:
        print(f"✗ Error during real data test: {e}")
        raise


def test_loss_computation():
    """Test that the model outputs can be used with standard loss functions."""
    print("\nTesting loss computation...")
    
    model = EnhancedMVFoulsModel()
    
    # Create dummy data
    batch_size = 3
    num_clips = 4
    dummy_videos = [
        torch.randn(num_clips, 3, 16, 224, 224) for _ in range(batch_size)
    ]
    
    # Create dummy labels
    action_labels = torch.randint(0, 8, (batch_size,))
    severity_labels = torch.randint(0, 4, (batch_size,))
    
    # Forward pass
    action_logits, severity_logits, confidence_scores, attention_info = model(dummy_videos)
    
    # Test loss computation
    criterion = torch.nn.CrossEntropyLoss()
    
    action_loss = criterion(action_logits, action_labels)
    severity_loss = criterion(severity_logits, severity_labels)
    diversity_loss = attention_info['diversity_loss']
    
    total_loss = action_loss + severity_loss + diversity_loss
    
    print(f"✓ Action loss: {action_loss.item():.4f}")
    print(f"✓ Severity loss: {severity_loss.item():.4f}")
    print(f"✓ Diversity loss: {diversity_loss.item():.4f}")
    print(f"✓ Total loss: {total_loss.item():.4f}")
    
    # Test backward pass
    total_loss.backward()
    print("✓ Backward pass successful")


if __name__ == "__main__":
    test_enhanced_model_with_real_data()
    test_loss_computation()
    print("\n🎉 All tests completed successfully!")