# Design Document

## Overview

This design addresses the critical training issues in the MVFouls model by implementing a multi-phase optimization strategy. The current model suffers from severe class imbalance problems, with most minority classes achieving 0% recall and validation loss exceeding 12. The solution involves progressive training strategies, architecture simplification options, advanced rebalancing techniques, and comprehensive monitoring.

## Architecture

### Current Issues Analysis
- **Dual Attention Complexity**: Separate attention mechanisms for action/severity may be over-parameterized
- **Aggressive Rebalancing**: Current weights (up to 10x) and sampling (8x for Red Card) cause training instability
- **Learning Rate Mismatch**: Backbone at 0.1x LR may be too constrained for fine-tuning
- **Loss Function Complexity**: Focal loss with extreme parameters may hinder convergence

### Proposed Architecture Variants

#### Variant A: Simplified Single-Task Architecture
```python
class SimplifiedMVFoulsModel(nn.Module):
    def __init__(self):
        # Single attention mechanism
        # Shared feature extraction
        # Task-specific heads only at final layer
```

#### Variant B: Progressive Complexity Architecture
```python
class ProgressiveMVFoulsModel(nn.Module):
    def __init__(self, complexity_level=1):
        # Level 1: Basic shared features
        # Level 2: Add attention
        # Level 3: Add task-specific processing
```

#### Variant C: Optimized Current Architecture
```python
class OptimizedMVFoulsModel(nn.Module):
    def __init__(self):
        # Reduced attention heads (4 instead of 8)
        # Simplified feature processing
        # Better regularization strategy
```

## Components and Interfaces

### 1. Progressive Training Manager
```python
class ProgressiveTrainingManager:
    def __init__(self, model, train_loader, val_loader):
        self.phases = [
            Phase1_BasicTraining(),
            Phase2_BalancedTraining(), 
            Phase3_AdvancedTraining()
        ]
    
    def train_progressive(self):
        # Phase 1: Train on balanced subset
        # Phase 2: Introduce full dataset with moderate rebalancing
        # Phase 3: Fine-tune with advanced techniques
```

### 2. Advanced Rebalancing System
```python
class SmartRebalancer:
    def __init__(self, class_frequencies):
        self.calculate_moderate_weights()
        self.setup_curriculum_sampling()
    
    def get_phase_weights(self, phase):
        # Phase 1: Minimal rebalancing (1.5x max)
        # Phase 2: Moderate rebalancing (3x max)  
        # Phase 3: Targeted rebalancing for remaining issues
```

### 3. Curriculum Learning System
```python
class CurriculumLearner:
    def __init__(self, dataset):
        self.easy_samples = self.identify_easy_samples()
        self.hard_samples = self.identify_hard_samples()
    
    def get_curriculum_loader(self, epoch):
        # Start with easy samples
        # Gradually introduce harder samples
        # Ensure minority class representation
```

### 4. Enhanced Monitoring System
```python
class TrainingMonitor:
    def __init__(self):
        self.metrics_tracker = MetricsTracker()
        self.visualization = VisualizationManager()
    
    def log_detailed_metrics(self, predictions, labels):
        # Per-class precision/recall/f1
        # Confusion matrices
        # Learning curves
        # Class distribution analysis
```

## Data Models

### Training Configuration
```python
@dataclass
class TrainingConfig:
    # Phase-specific parameters
    phase1_epochs: int = 10
    phase2_epochs: int = 15  
    phase3_epochs: int = 15
    
    # Learning rates per phase
    phase1_lr: float = 1e-4
    phase2_lr: float = 5e-5
    phase3_lr: float = 2e-5
    
    # Rebalancing parameters
    max_weight_phase1: float = 1.5
    max_weight_phase2: float = 3.0
    max_weight_phase3: float = 5.0
    
    # Architecture variant
    model_variant: str = "simplified"  # "simplified", "progressive", "optimized"
```

### Metrics Tracking
```python
@dataclass
class DetailedMetrics:
    epoch: int
    phase: str
    
    # Overall metrics
    train_loss: float
    val_loss: float
    
    # Per-task metrics
    action_accuracy: float
    action_macro_f1: float
    action_per_class_recall: List[float]
    
    severity_accuracy: float
    severity_macro_f1: float
    severity_per_class_recall: List[float]
    
    # Class-specific details
    confusion_matrices: Dict[str, np.ndarray]
    learning_curves: Dict[str, List[float]]
```

## Error Handling

### Training Stability Monitoring
- **Loss Explosion Detection**: Monitor for sudden loss spikes > 2x previous value
- **Gradient Monitoring**: Track gradient norms to detect vanishing/exploding gradients
- **Class Collapse Detection**: Alert when any class drops to 0% recall for 3+ consecutive epochs
- **Memory Management**: Handle CUDA OOM errors gracefully with batch size reduction

### Recovery Strategies
- **Automatic Learning Rate Reduction**: Reduce LR by 0.5x when loss spikes detected
- **Checkpoint Rollback**: Revert to previous stable checkpoint on training instability
- **Dynamic Batch Size**: Reduce batch size if memory issues occur
- **Fallback Architecture**: Switch to simpler model variant if complex version fails

## Testing Strategy

### A/B Testing Framework
1. **Baseline Comparison**: Current model vs. simplified variants
2. **Progressive Training**: Compare single-phase vs. multi-phase training
3. **Rebalancing Strategies**: Compare different weight/sampling approaches
4. **Architecture Variants**: Systematic comparison of model complexities

### Evaluation Metrics
- **Primary**: Macro recall for both tasks (target: >45%)
- **Secondary**: Macro F1-score for both tasks (target: >0.40)
- **Tertiary**: Per-class recall (target: >25% for all classes)
- **Stability**: Validation loss convergence (target: <5.0)
- **Efficiency**: Training time and memory usage

### Test Scenarios
1. **Minority Class Focus**: Specific evaluation on Red Card and rare actions
2. **Generalization**: Cross-validation with different train/val splits
3. **Robustness**: Performance under different random seeds
4. **Scalability**: Performance with different batch sizes and accumulation steps

## Implementation Phases

### Phase 1: Foundation (Epochs 1-10)
- Use simplified architecture or reduced complexity
- Train on class-balanced subset (equal samples per class)
- Minimal data augmentation
- Conservative learning rates
- **Goal**: Establish basic learning patterns for all classes

### Phase 2: Scaling (Epochs 11-25)
- Introduce full dataset with moderate rebalancing
- Add progressive data augmentation
- Implement curriculum learning
- **Goal**: Scale to full dataset while maintaining class balance

### Phase 3: Optimization (Epochs 26-40)
- Fine-tune with advanced techniques
- Apply targeted improvements for remaining weak classes
- Implement ensemble or pseudo-labeling if needed
- **Goal**: Achieve target performance metrics

## Key Design Decisions

### 1. Progressive Training Over Aggressive Rebalancing
**Rationale**: Current 10x weights and 8x sampling cause instability. Progressive approach allows model to learn gradually.

### 2. Architecture Simplification Options
**Rationale**: Current dual-attention system may be over-complex for dataset size. Provide simpler alternatives.

### 3. Phase-Based Learning Rates
**Rationale**: Different phases need different learning dynamics. Start higher for basic learning, reduce for fine-tuning.

### 4. Comprehensive Monitoring
**Rationale**: Current training lacks visibility into per-class behavior. Detailed monitoring enables informed decisions.

### 5. Curriculum Learning Integration
**Rationale**: Learning easy examples first, then hard ones, often improves final performance on imbalanced datasets.

This design provides a systematic approach to address the training issues while maintaining flexibility to adapt based on experimental results.