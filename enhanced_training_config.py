"""
Enhanced Training Configuration Management System

This module provides comprehensive configuration management for the enhanced
training pipeline, including validation, defaults, and environment-specific
adjustments.

Requirements: Comprehensive configuration management and error handling
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    folder: str = "mvfouls"
    train_split: str = "train"
    val_split: str = "test"
    start_frame: int = 63
    end_frame: int = 86
    
    def validate(self) -> List[str]:
        """Validate data configuration"""
        errors = []
        
        if not os.path.exists(self.folder):
            errors.append(f"Data folder '{self.folder}' does not exist")
        
        if self.start_frame >= self.end_frame:
            errors.append(f"start_frame ({self.start_frame}) must be less than end_frame ({self.end_frame})")
        
        if self.start_frame < 0 or self.end_frame < 0:
            errors.append("Frame indices must be non-negative")
        
        return errors


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    use_enhanced: bool = True
    aggregation: str = "attention"
    input_size: List[int] = field(default_factory=lambda: [224, 224])
    num_action_classes: int = 8
    num_severity_classes: int = 4
    dropout_rate: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        if self.aggregation not in ["max", "mean", "attention"]:
            errors.append(f"Invalid aggregation method: {self.aggregation}")
        
        if len(self.input_size) != 2:
            errors.append("input_size must be a list of 2 integers [height, width]")
        
        if any(size <= 0 for size in self.input_size):
            errors.append("input_size dimensions must be positive")
        
        if self.num_action_classes <= 0 or self.num_severity_classes <= 0:
            errors.append("Number of classes must be positive")
        
        if not 0.0 <= self.dropout_rate <= 1.0:
            errors.append("dropout_rate must be between 0.0 and 1.0")
        
        return errors


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    accumulation_steps: int = 4
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate training configuration"""
        errors = []
        
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        
        if self.num_workers < 0:
            errors.append("num_workers must be non-negative")
        
        if self.accumulation_steps <= 0:
            errors.append("accumulation_steps must be positive")
        
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        
        if self.gradient_clip_norm <= 0:
            errors.append("gradient_clip_norm must be positive")
        
        return errors
    
    def adjust_for_environment(self):
        """Adjust configuration based on environment"""
        # Adjust for Windows
        if os.name == 'nt' and self.num_workers > 0:
            logger.warning("Setting num_workers to 0 for Windows compatibility")
            self.num_workers = 0
        
        # Adjust for available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_memory < 8 and self.batch_size > 2:
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is limited, reducing batch_size to 2")
                self.batch_size = 2
                self.accumulation_steps = max(4, self.accumulation_steps * 2)  # Compensate with accumulation


@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced data pipeline"""
    enabled: bool = True
    synthetic_generation: Dict[str, Any] = field(default_factory=lambda: {
        'target_classes': {
            'action': {4: 50, 7: 50},  # Pushing: 50, Dive: 50
            'severity': {3: 100}       # Red Card: 100
        },
        'mixup_alpha': 0.4,
        'temporal_alpha': 0.3,
        'quality_threshold': 0.7
    })
    quality_validation: Dict[str, Any] = field(default_factory=lambda: {
        'feature_similarity_threshold': 0.7,
        'temporal_consistency_threshold': 0.8,
        'label_consistency_threshold': 0.9
    })
    stratified_sampling: Dict[str, Any] = field(default_factory=lambda: {
        'min_minority_per_batch': 1,
        'minority_threshold': 0.05
    })
    advanced_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        'temporal_jitter_range': [0.8, 1.2],
        'spatial_jitter_prob': 0.5,
        'color_jitter_prob': 0.3,
        'gaussian_noise_prob': 0.2,
        'frame_dropout_prob': 0.1
    })
    
    def validate(self) -> List[str]:
        """Validate enhanced pipeline configuration"""
        errors = []
        
        if not isinstance(self.synthetic_generation, dict):
            errors.append("synthetic_generation must be a dictionary")
        
        if not isinstance(self.stratified_sampling, dict):
            errors.append("stratified_sampling must be a dictionary")
        
        if not isinstance(self.advanced_augmentation, dict):
            errors.append("advanced_augmentation must be a dictionary")
        
        # Validate synthetic generation parameters
        if 'target_classes' in self.synthetic_generation:
            target_classes = self.synthetic_generation['target_classes']
            if not isinstance(target_classes, dict):
                errors.append("target_classes must be a dictionary")
            else:
                for task, classes in target_classes.items():
                    if task not in ['action', 'severity']:
                        errors.append(f"Invalid task in target_classes: {task}")
                    if not isinstance(classes, dict):
                        errors.append(f"Classes for {task} must be a dictionary")
                    else:
                        for class_id, count in classes.items():
                            if not isinstance(class_id, int) or class_id < 0:
                                errors.append(f"Invalid class_id in {task}: {class_id}")
                            if not isinstance(count, int) or count <= 0:
                                errors.append(f"Invalid count for {task} class {class_id}: {count}")
        
        return errors


@dataclass
class AdaptiveLossConfig:
    """Configuration for adaptive loss system"""
    enabled: bool = True
    initial_gamma: float = 2.0
    initial_alpha: float = 1.0
    min_gamma: float = 0.5
    max_gamma: float = 5.0
    min_alpha: float = 0.1
    max_alpha: float = 10.0
    recall_threshold: float = 0.1
    weight_increase_factor: float = 1.5
    adaptation_rate: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate adaptive loss configuration"""
        errors = []
        
        if self.initial_gamma <= 0:
            errors.append("initial_gamma must be positive")
        
        if self.initial_alpha <= 0:
            errors.append("initial_alpha must be positive")
        
        if self.min_gamma >= self.max_gamma:
            errors.append("min_gamma must be less than max_gamma")
        
        if self.min_alpha >= self.max_alpha:
            errors.append("min_alpha must be less than max_alpha")
        
        if not 0.0 <= self.recall_threshold <= 1.0:
            errors.append("recall_threshold must be between 0.0 and 1.0")
        
        if self.weight_increase_factor <= 1.0:
            errors.append("weight_increase_factor must be greater than 1.0")
        
        if not 0.0 <= self.adaptation_rate <= 1.0:
            errors.append("adaptation_rate must be between 0.0 and 1.0")
        
        return errors


@dataclass
class CurriculumLearningConfig:
    """Configuration for curriculum learning"""
    enabled: bool = True
    easy_stage_epochs: int = 10
    medium_stage_epochs: int = 8
    hard_stage_epochs: int = 7
    transition_smoothing: float = 0.1
    performance_threshold: float = 0.3
    
    def validate(self) -> List[str]:
        """Validate curriculum learning configuration"""
        errors = []
        
        if self.easy_stage_epochs <= 0:
            errors.append("easy_stage_epochs must be positive")
        
        if self.medium_stage_epochs <= 0:
            errors.append("medium_stage_epochs must be positive")
        
        if self.hard_stage_epochs <= 0:
            errors.append("hard_stage_epochs must be positive")
        
        if not 0.0 <= self.transition_smoothing <= 1.0:
            errors.append("transition_smoothing must be between 0.0 and 1.0")
        
        if not 0.0 <= self.performance_threshold <= 1.0:
            errors.append("performance_threshold must be between 0.0 and 1.0")
        
        return errors


@dataclass
class PerformanceMonitoringConfig:
    """Configuration for performance monitoring"""
    enabled: bool = True
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'zero_recall_epochs': 3,
        'performance_drop_threshold': 0.1,
        'plateau_epochs': 5,
        'min_acceptable_recall': 0.05,
        'target_combined_recall': 0.45,
        'gradient_norm_threshold': 10.0,
        'loss_divergence_threshold': 2.0
    })
    alert_frequency: int = 1  # Check alerts every N epochs
    save_plots: bool = True
    
    def validate(self) -> List[str]:
        """Validate performance monitoring configuration"""
        errors = []
        
        if not isinstance(self.thresholds, dict):
            errors.append("thresholds must be a dictionary")
        
        required_thresholds = [
            'zero_recall_epochs', 'performance_drop_threshold', 'plateau_epochs',
            'min_acceptable_recall', 'target_combined_recall', 'gradient_norm_threshold',
            'loss_divergence_threshold'
        ]
        
        for threshold in required_thresholds:
            if threshold not in self.thresholds:
                errors.append(f"Missing required threshold: {threshold}")
            elif not isinstance(self.thresholds[threshold], (int, float)):
                errors.append(f"Threshold {threshold} must be a number")
            elif self.thresholds[threshold] <= 0:
                errors.append(f"Threshold {threshold} must be positive")
        
        if self.alert_frequency <= 0:
            errors.append("alert_frequency must be positive")
        
        return errors


@dataclass
class EnsembleSystemConfig:
    """Configuration for ensemble system"""
    enabled: bool = True
    max_checkpoints: int = 8
    min_performance_threshold: float = 0.35
    diversity_weight: float = 0.2
    voting_method: str = "confidence_weighted"
    
    def validate(self) -> List[str]:
        """Validate ensemble system configuration"""
        errors = []
        
        if self.max_checkpoints <= 0:
            errors.append("max_checkpoints must be positive")
        
        if not 0.0 <= self.min_performance_threshold <= 1.0:
            errors.append("min_performance_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.diversity_weight <= 1.0:
            errors.append("diversity_weight must be between 0.0 and 1.0")
        
        if self.voting_method not in ["simple", "weighted", "confidence_weighted"]:
            errors.append(f"Invalid voting_method: {self.voting_method}")
        
        return errors


@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""
    enabled: bool = False  # Disabled by default due to computational cost
    optimization_strategy: str = "bayesian_gp"
    n_calls: int = 20
    evaluation_epochs: int = 3
    enable_adaptive: bool = True
    enable_sensitivity: bool = True
    
    def validate(self) -> List[str]:
        """Validate hyperparameter optimization configuration"""
        errors = []
        
        valid_strategies = ["bayesian_gp", "bayesian_rf", "bayesian_gbrt", "random_search", "adaptive_only"]
        if self.optimization_strategy not in valid_strategies:
            errors.append(f"Invalid optimization_strategy: {self.optimization_strategy}")
        
        if self.n_calls <= 0:
            errors.append("n_calls must be positive")
        
        if self.evaluation_epochs <= 0:
            errors.append("evaluation_epochs must be positive")
        
        return errors


@dataclass
class EnhancedTrainingConfig:
    """Complete configuration for enhanced training system"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    enhanced_pipeline: EnhancedPipelineConfig = field(default_factory=EnhancedPipelineConfig)
    adaptive_loss: AdaptiveLossConfig = field(default_factory=AdaptiveLossConfig)
    curriculum_learning: CurriculumLearningConfig = field(default_factory=CurriculumLearningConfig)
    performance_monitoring: PerformanceMonitoringConfig = field(default_factory=PerformanceMonitoringConfig)
    ensemble_system: EnsembleSystemConfig = field(default_factory=EnsembleSystemConfig)
    hyperparameter_optimization: HyperparameterOptimizationConfig = field(default_factory=HyperparameterOptimizationConfig)
    
    # Global settings
    save_dir: str = "enhanced_training_results"
    experiment_name: str = "mvfouls_enhanced_training"
    random_seed: int = 42
    
    def validate(self) -> List[str]:
        """Validate complete configuration"""
        all_errors = []
        
        # Validate each component
        components = [
            ('data', self.data),
            ('model', self.model),
            ('training', self.training),
            ('enhanced_pipeline', self.enhanced_pipeline),
            ('adaptive_loss', self.adaptive_loss),
            ('curriculum_learning', self.curriculum_learning),
            ('performance_monitoring', self.performance_monitoring),
            ('ensemble_system', self.ensemble_system),
            ('hyperparameter_optimization', self.hyperparameter_optimization)
        ]
        
        for component_name, component in components:
            component_errors = component.validate()
            for error in component_errors:
                all_errors.append(f"{component_name}: {error}")
        
        # Cross-component validation
        total_curriculum_epochs = (
            self.curriculum_learning.easy_stage_epochs +
            self.curriculum_learning.medium_stage_epochs +
            self.curriculum_learning.hard_stage_epochs
        )
        
        if self.curriculum_learning.enabled and total_curriculum_epochs > self.training.epochs:
            all_errors.append(
                f"Total curriculum epochs ({total_curriculum_epochs}) exceeds training epochs ({self.training.epochs})"
            )
        
        # Validate save directory
        if not self.save_dir:
            all_errors.append("save_dir cannot be empty")
        
        if not self.experiment_name:
            all_errors.append("experiment_name cannot be empty")
        
        return all_errors
    
    def adjust_for_environment(self):
        """Adjust configuration for current environment"""
        logger.info("Adjusting configuration for current environment...")
        
        # Adjust training config
        self.training.adjust_for_environment()
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        logger.info("Environment adjustments completed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedTrainingConfig':
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Create config object from dictionary
        config = cls()
        config._update_from_dict(config_dict)
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if hasattr(attr, '__dict__'):  # It's a dataclass
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(attr, sub_key):
                                setattr(attr, sub_key, sub_value)
                else:
                    setattr(self, key, value)


class ConfigurationManager:
    """Manager for configuration validation and environment setup"""
    
    def __init__(self, config: Optional[EnhancedTrainingConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Configuration object (creates default if None)
        """
        self.config = config or EnhancedTrainingConfig()
        self.validation_errors = []
        self.warnings = []
    
    def validate_configuration(self) -> bool:
        """
        Validate the complete configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Validating configuration...")
        
        self.validation_errors = self.config.validate()
        
        if self.validation_errors:
            logger.error("Configuration validation failed:")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def setup_environment(self):
        """Set up environment based on configuration"""
        logger.info("Setting up environment...")
        
        # Adjust configuration for environment
        self.config.adjust_for_environment()
        
        # Set up logging
        log_dir = os.path.join(self.config.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file logging
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{self.config.experiment_name}.log")
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        
        logger.info("Environment setup completed")
    
    def create_experiment_config(self, 
                                experiment_name: str,
                                modifications: Optional[Dict[str, Any]] = None) -> EnhancedTrainingConfig:
        """
        Create a new experiment configuration with modifications.
        
        Args:
            experiment_name: Name for the experiment
            modifications: Dictionary of configuration modifications
            
        Returns:
            New configuration object
        """
        # Create a copy of the current configuration
        new_config_dict = self.config.to_dict()
        
        # Apply modifications
        if modifications:
            self._apply_modifications(new_config_dict, modifications)
        
        # Create new config object
        new_config = EnhancedTrainingConfig()
        new_config._update_from_dict(new_config_dict)
        new_config.experiment_name = experiment_name
        new_config.save_dir = os.path.join(new_config.save_dir, experiment_name)
        
        return new_config
    
    def _apply_modifications(self, config_dict: Dict[str, Any], modifications: Dict[str, Any]):
        """Apply modifications to configuration dictionary"""
        for key, value in modifications.items():
            if '.' in key:
                # Handle nested keys like 'training.learning_rate'
                keys = key.split('.')
                current = config_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config_dict[key] = value
    
    def get_config_summary(self) -> str:
        """Get a summary of the current configuration"""
        summary = []
        summary.append(f"Experiment: {self.config.experiment_name}")
        summary.append(f"Save Directory: {self.config.save_dir}")
        summary.append(f"Random Seed: {self.config.random_seed}")
        summary.append("")
        
        # Training configuration
        summary.append("Training Configuration:")
        summary.append(f"  Epochs: {self.config.training.epochs}")
        summary.append(f"  Batch Size: {self.config.training.batch_size}")
        summary.append(f"  Learning Rate: {self.config.training.learning_rate}")
        summary.append(f"  Mixed Precision: {self.config.training.mixed_precision}")
        summary.append("")
        
        # Component status
        summary.append("Component Status:")
        summary.append(f"  Enhanced Pipeline: {'Enabled' if self.config.enhanced_pipeline.enabled else 'Disabled'}")
        summary.append(f"  Adaptive Loss: {'Enabled' if self.config.adaptive_loss.enabled else 'Disabled'}")
        summary.append(f"  Curriculum Learning: {'Enabled' if self.config.curriculum_learning.enabled else 'Disabled'}")
        summary.append(f"  Performance Monitoring: {'Enabled' if self.config.performance_monitoring.enabled else 'Disabled'}")
        summary.append(f"  Ensemble System: {'Enabled' if self.config.ensemble_system.enabled else 'Disabled'}")
        summary.append(f"  Hyperparameter Optimization: {'Enabled' if self.config.hyperparameter_optimization.enabled else 'Disabled'}")
        
        return "\n".join(summary)


def create_default_config() -> EnhancedTrainingConfig:
    """Create default configuration for enhanced training"""
    return EnhancedTrainingConfig()


def create_quick_test_config() -> EnhancedTrainingConfig:
    """Create configuration for quick testing"""
    config = EnhancedTrainingConfig()
    
    # Reduce training time for testing
    config.training.epochs = 3
    config.training.batch_size = 2
    config.curriculum_learning.easy_stage_epochs = 1
    config.curriculum_learning.medium_stage_epochs = 1
    config.curriculum_learning.hard_stage_epochs = 1
    
    # Disable expensive components
    config.hyperparameter_optimization.enabled = False
    config.ensemble_system.enabled = False
    
    # Reduce synthetic data generation
    config.enhanced_pipeline.synthetic_generation['target_classes'] = {
        'action': {4: 5, 7: 5},
        'severity': {3: 10}
    }
    
    config.experiment_name = "quick_test"
    config.save_dir = "test_results"
    
    return config


def create_production_config() -> EnhancedTrainingConfig:
    """Create configuration optimized for production training"""
    config = EnhancedTrainingConfig()
    
    # Production training settings
    config.training.epochs = 50
    config.training.batch_size = 8
    config.training.learning_rate = 5e-5
    config.training.early_stopping_patience = 15
    
    # Enable all components
    config.enhanced_pipeline.enabled = True
    config.adaptive_loss.enabled = True
    config.curriculum_learning.enabled = True
    config.performance_monitoring.enabled = True
    config.ensemble_system.enabled = True
    config.hyperparameter_optimization.enabled = False  # Still expensive
    
    # Increase synthetic data generation
    config.enhanced_pipeline.synthetic_generation['target_classes'] = {
        'action': {4: 100, 7: 100},
        'severity': {3: 200}
    }
    
    config.experiment_name = "production_training"
    config.save_dir = "production_results"
    
    return config


if __name__ == "__main__":
    print("Testing Enhanced Training Configuration Management...")
    
    # Test default configuration
    config = create_default_config()
    manager = ConfigurationManager(config)
    
    # Validate configuration
    is_valid = manager.validate_configuration()
    print(f"✓ Default configuration valid: {is_valid}")
    
    # Test configuration summary
    summary = manager.get_config_summary()
    print(f"✓ Configuration summary generated ({len(summary.split())} words)")
    
    # Test saving and loading
    config_path = "test_config.json"
    config.save(config_path)
    print(f"✓ Configuration saved to {config_path}")
    
    loaded_config = EnhancedTrainingConfig.load(config_path)
    print(f"✓ Configuration loaded successfully")
    
    # Test quick test configuration
    test_config = create_quick_test_config()
    test_manager = ConfigurationManager(test_config)
    test_valid = test_manager.validate_configuration()
    print(f"✓ Quick test configuration valid: {test_valid}")
    
    # Test production configuration
    prod_config = create_production_config()
    prod_manager = ConfigurationManager(prod_config)
    prod_valid = prod_manager.validate_configuration()
    print(f"✓ Production configuration valid: {prod_valid}")
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)
    
    print("\n✓ All configuration management tests passed!")