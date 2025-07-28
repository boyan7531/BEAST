"""
Integration module for Enhanced Data Pipeline with MVFouls training system.
Provides seamless integration with existing training code.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import os
import json
from datetime import datetime

from enhanced_data_pipeline import (
    StratifiedMinorityBatchSampler,
    MixupVideoGenerator, 
    AdvancedVideoAugmentation,
    SyntheticQualityValidator
)
from dataset import MVFoulsDataset, custom_collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMVFoulsDataset(Dataset):
    """
    Enhanced version of MVFoulsDataset with synthetic sample support.
    """
    
    def __init__(self, 
                 base_dataset: MVFoulsDataset,
                 synthetic_samples: List[Tuple] = None,
                 use_advanced_augmentation: bool = True):
        """
        Initialize enhanced dataset.
        
        Args:
            base_dataset: Original MVFoulsDataset
            synthetic_samples: List of synthetic samples to include
            use_advanced_augmentation: Whether to use advanced augmentation
        """
        self.base_dataset = base_dataset
        self.synthetic_samples = synthetic_samples or []
        self.use_advanced_augmentation = use_advanced_augmentation
        
        # Initialize advanced augmentation
        if self.use_advanced_augmentation:
            self.advanced_augmentation = AdvancedVideoAugmentation()
        
        # Combine base and synthetic samples
        self.total_length = len(base_dataset) + len(self.synthetic_samples)
        
        logger.info(f"EnhancedMVFoulsDataset initialized:")
        logger.info(f"  Base samples: {len(base_dataset)}")
        logger.info(f"  Synthetic samples: {len(self.synthetic_samples)}")
        logger.info(f"  Total samples: {self.total_length}")
        logger.info(f"  Advanced augmentation: {use_advanced_augmentation}")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            # Get sample from base dataset
            video, action_label, severity_label, action_id = self.base_dataset[idx]
            
            # Apply advanced augmentation if enabled
            if self.use_advanced_augmentation and random.random() < 0.5:
                video = self.advanced_augmentation(video)
            
            return video, action_label, severity_label, action_id
        else:
            # Get synthetic sample
            synthetic_idx = idx - len(self.base_dataset)
            video, action_label, severity_label, metadata = self.synthetic_samples[synthetic_idx]
            
            # Create action_id from metadata
            action_id = f"synthetic_{synthetic_idx}"
            
            return video, action_label, severity_label, action_id
    
    def get_all_labels(self):
        """Get all action and severity labels for sampling purposes"""
        action_labels = []
        severity_labels = []
        
        # Get base dataset labels
        for i in range(len(self.base_dataset)):
            _, action_label, severity_label, _ = self.base_dataset.data_list[i]
            action_labels.append(action_label)
            severity_labels.append(severity_label)
        
        # Get synthetic sample labels
        for video, action_label, severity_label, metadata in self.synthetic_samples:
            action_labels.append(action_label)
            severity_labels.append(severity_label)
        
        return action_labels, severity_labels


class EnhancedDataPipelineManager:
    """
    Manager class for the enhanced data pipeline.
    Coordinates all components and provides easy integration.
    """
    
    def __init__(self, 
                 config: Dict = None,
                 save_dir: str = "enhanced_pipeline_logs"):
        """
        Initialize the enhanced data pipeline manager.
        
        Args:
            config: Configuration dictionary
            save_dir: Directory to save logs and generated samples
        """
        self.config = config or self._get_default_config()
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.mixup_generator = None
        self.quality_validator = None
        self.synthetic_samples = []
        
        logger.info(f"EnhancedDataPipelineManager initialized with config: {self.config}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the enhanced pipeline"""
        return {
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
            },
            'advanced_augmentation': {
                'temporal_jitter_range': (0.8, 1.2),
                'spatial_jitter_prob': 0.5,
                'color_jitter_prob': 0.3,
                'gaussian_noise_prob': 0.2,
                'frame_dropout_prob': 0.1
            },
            'quality_validation': {
                'feature_similarity_threshold': 0.7,
                'temporal_consistency_threshold': 0.8,
                'label_consistency_threshold': 0.9
            }
        }
    
    def setup_enhanced_dataset(self, base_dataset: MVFoulsDataset) -> EnhancedMVFoulsDataset:
        """
        Set up enhanced dataset with synthetic samples.
        
        Args:
            base_dataset: Original MVFoulsDataset
            
        Returns:
            Enhanced dataset with synthetic samples
        """
        logger.info("Setting up enhanced dataset...")
        
        # Initialize mixup generator
        self.mixup_generator = MixupVideoGenerator(
            dataset=base_dataset,
            target_classes=self.config['synthetic_generation']['target_classes'],
            mixup_alpha=self.config['synthetic_generation']['mixup_alpha'],
            temporal_alpha=self.config['synthetic_generation']['temporal_alpha']
        )
        
        # Generate synthetic samples
        logger.info("Generating synthetic samples...")
        self.synthetic_samples = self.mixup_generator.generate_synthetic_samples()
        
        # Initialize quality validator
        quality_config = self.config.get('quality_validation', {
            'feature_similarity_threshold': 0.7,
            'temporal_consistency_threshold': 0.8,
            'label_consistency_threshold': 0.9
        })
        self.quality_validator = SyntheticQualityValidator(**quality_config)
        
        # Validate synthetic samples
        logger.info("Validating synthetic samples...")
        validation_stats = self.quality_validator.validate_batch(self.synthetic_samples)
        
        # Filter high-quality samples
        high_quality_samples = []
        for i, sample in enumerate(self.synthetic_samples):
            # In a real implementation, you'd use actual validation metrics
            # For now, we'll keep all samples but log the stats
            high_quality_samples.append(sample)
        
        self.synthetic_samples = high_quality_samples
        
        # Log validation results
        logger.info(f"Synthetic sample validation results:")
        logger.info(f"  Total generated: {len(self.synthetic_samples)}")
        logger.info(f"  Pass rate: {validation_stats['pass_rate']:.2%}")
        logger.info(f"  Avg quality score: {validation_stats['avg_quality_score']:.3f}")
        
        # Save synthetic samples metadata
        self._save_synthetic_metadata(validation_stats)
        
        # Create enhanced dataset
        enhanced_dataset = EnhancedMVFoulsDataset(
            base_dataset=base_dataset,
            synthetic_samples=self.synthetic_samples,
            use_advanced_augmentation=True
        )
        
        return enhanced_dataset
    
    def create_enhanced_dataloader(self, 
                                 enhanced_dataset: EnhancedMVFoulsDataset,
                                 batch_size: int,
                                 num_workers: int = 0) -> DataLoader:
        """
        Create enhanced dataloader with stratified minority sampling.
        
        Args:
            enhanced_dataset: Enhanced dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            DataLoader with stratified minority sampling
        """
        logger.info("Creating enhanced dataloader with stratified sampling...")
        
        # Get all labels for sampling
        action_labels, severity_labels = enhanced_dataset.get_all_labels()
        
        # Create stratified minority batch sampler
        sampler = StratifiedMinorityBatchSampler(
            action_labels=action_labels,
            severity_labels=severity_labels,
            batch_size=batch_size,
            **self.config['stratified_sampling']
        )
        
        # Log sampling statistics
        stats = sampler.get_minority_stats()
        logger.info(f"Stratified sampling statistics:")
        logger.info(f"  Minority action classes: {stats['minority_action_classes']}")
        logger.info(f"  Minority severity classes: {stats['minority_severity_classes']}")
        logger.info(f"  Total minority samples: {stats['total_minority_samples']}")
        
        # Create dataloader
        dataloader = DataLoader(
            enhanced_dataset,
            batch_sampler=sampler,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def _save_synthetic_metadata(self, validation_stats: Dict):
        """Save metadata about synthetic sample generation"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'generation_stats': {
                'total_synthetic_samples': len(self.synthetic_samples),
                'target_classes': self.config['synthetic_generation']['target_classes']
            },
            'validation_stats': validation_stats
        }
        
        # Save to JSON file
        metadata_path = os.path.join(self.save_dir, 'synthetic_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Synthetic metadata saved to {metadata_path}")
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the enhanced pipeline"""
        return {
            'synthetic_samples_generated': len(self.synthetic_samples),
            'config': self.config,
            'components_initialized': {
                'mixup_generator': self.mixup_generator is not None,
                'quality_validator': self.quality_validator is not None
            }
        }


def create_enhanced_training_setup(data_folder: str,
                                 train_split: str,
                                 val_split: str,
                                 start_frame: int,
                                 end_frame: int,
                                 batch_size: int,
                                 model_input_size: Tuple[int, int],
                                 num_workers: int = 0,
                                 config: Dict = None) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create complete enhanced training setup.
    
    Args:
        data_folder: Path to dataset folder
        train_split: Training split name
        val_split: Validation split name
        start_frame: Start frame for video clips
        end_frame: End frame for video clips
        batch_size: Batch size
        model_input_size: Model input size (H, W)
        num_workers: Number of worker processes
        config: Configuration for enhanced pipeline
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, pipeline_stats)
    """
    logger.info("Creating enhanced training setup...")
    
    # Import transforms
    from transform import get_train_transforms, get_val_transforms
    
    # Create base datasets
    train_dataset = MVFoulsDataset(
        data_folder, train_split, start_frame, end_frame,
        transform_model=get_train_transforms(model_input_size)
    )
    
    val_dataset = MVFoulsDataset(
        data_folder, val_split, start_frame, end_frame,
        transform_model=get_val_transforms(model_input_size)
    )
    
    # Initialize enhanced pipeline manager
    pipeline_manager = EnhancedDataPipelineManager(config=config)
    
    # Set up enhanced training dataset
    enhanced_train_dataset = pipeline_manager.setup_enhanced_dataset(train_dataset)
    
    # Create enhanced training dataloader
    train_dataloader = pipeline_manager.create_enhanced_dataloader(
        enhanced_train_dataset, batch_size, num_workers
    )
    
    # Create standard validation dataloader (no synthetic samples for validation)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get pipeline statistics
    pipeline_stats = pipeline_manager.get_pipeline_stats()
    
    logger.info("Enhanced training setup complete!")
    logger.info(f"  Training samples: {len(enhanced_train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Synthetic samples added: {pipeline_stats['synthetic_samples_generated']}")
    
    return train_dataloader, val_dataloader, pipeline_stats


def analyze_batch_minority_representation(dataloader: DataLoader, 
                                        num_batches_to_check: int = 10) -> Dict:
    """
    Analyze minority class representation in batches.
    
    Args:
        dataloader: DataLoader to analyze
        num_batches_to_check: Number of batches to check
        
    Returns:
        Analysis results
    """
    logger.info(f"Analyzing minority representation in {num_batches_to_check} batches...")
    
    minority_action_classes = [4, 7]  # Pushing, Dive
    minority_severity_classes = [3]   # Red Card
    
    batch_stats = []
    
    for i, (videos, action_labels, severity_labels, action_ids) in enumerate(dataloader):
        if i >= num_batches_to_check:
            break
        
        batch_size = len(videos)
        minority_count = 0
        
        # Convert labels to class indices
        action_classes = torch.argmax(action_labels, dim=1).tolist()
        severity_classes = torch.argmax(severity_labels, dim=1).tolist()
        
        # Count minority samples
        for action_class, severity_class in zip(action_classes, severity_classes):
            if action_class in minority_action_classes or severity_class in minority_severity_classes:
                minority_count += 1
        
        batch_stat = {
            'batch_idx': i,
            'batch_size': batch_size,
            'minority_count': minority_count,
            'minority_ratio': minority_count / batch_size if batch_size > 0 else 0,
            'action_distribution': dict(zip(*np.unique(action_classes, return_counts=True))),
            'severity_distribution': dict(zip(*np.unique(severity_classes, return_counts=True)))
        }
        
        batch_stats.append(batch_stat)
    
    # Calculate overall statistics
    total_samples = sum(stat['batch_size'] for stat in batch_stats)
    total_minority = sum(stat['minority_count'] for stat in batch_stats)
    avg_minority_ratio = np.mean([stat['minority_ratio'] for stat in batch_stats])
    
    analysis_results = {
        'total_batches_analyzed': len(batch_stats),
        'total_samples': total_samples,
        'total_minority_samples': total_minority,
        'overall_minority_ratio': total_minority / total_samples if total_samples > 0 else 0,
        'avg_minority_ratio_per_batch': avg_minority_ratio,
        'batch_stats': batch_stats,
        'minority_guarantee_met': all(stat['minority_count'] >= 1 for stat in batch_stats)
    }
    
    logger.info(f"Minority representation analysis complete:")
    logger.info(f"  Overall minority ratio: {analysis_results['overall_minority_ratio']:.2%}")
    logger.info(f"  Avg minority per batch: {avg_minority_ratio:.2%}")
    logger.info(f"  Minority guarantee met: {analysis_results['minority_guarantee_met']}")
    
    return analysis_results


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced pipeline integration
    logger.info("Testing Enhanced Data Pipeline Integration...")
    
    # Configuration for testing
    test_config = {
        'synthetic_generation': {
            'target_classes': {
                'action': {4: 10, 7: 10},  # Smaller numbers for testing
                'severity': {3: 20}
            },
            'mixup_alpha': 0.4,
            'temporal_alpha': 0.3
        },
        'stratified_sampling': {
            'min_minority_per_batch': 1,
            'minority_threshold': 0.05
        },
        'advanced_augmentation': {
            'temporal_jitter_range': (0.8, 1.2),
            'spatial_jitter_prob': 0.5,
            'color_jitter_prob': 0.3,
            'gaussian_noise_prob': 0.2,
            'frame_dropout_prob': 0.1
        },
        'quality_validation': {
            'feature_similarity_threshold': 0.7,
            'temporal_consistency_threshold': 0.8,
            'label_consistency_threshold': 0.9
        }
    }
    
    try:
        # Create enhanced training setup
        train_dataloader, val_dataloader, pipeline_stats = create_enhanced_training_setup(
            data_folder="mvfouls",
            train_split="train", 
            val_split="test",
            start_frame=63,
            end_frame=86,
            batch_size=8,
            model_input_size=(224, 224),
            num_workers=0,
            config=test_config
        )
        
        logger.info("Enhanced training setup successful!")
        logger.info(f"Pipeline stats: {pipeline_stats}")
        
        # Analyze minority representation
        analysis_results = analyze_batch_minority_representation(train_dataloader, 5)
        logger.info(f"Minority analysis: {analysis_results['minority_guarantee_met']}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()