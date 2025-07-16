"""
Smart Rebalancing System for MVFouls Training

This module implements an adaptive rebalancing system that dynamically adjusts
sampling strategies, loss weights, and other parameters based on real-time
performance metrics to optimize training for imbalanced datasets.
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

class RebalancingStrategy(Enum):
    """Available rebalancing strategies"""
    WEIGHTED_SAMPLING = "weighted_sampling"
    FOCAL_LOSS = "focal_loss"
    CLASS_WEIGHTS = "class_weights"
    MIXUP = "mixup"
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    epoch: int
    overall_accuracy: float
    macro_f1: float
    macro_recall: float
    per_class_recall: Dict[int, float]
    per_class_precision: Dict[int, float]
    per_class_f1: Dict[int, float]
    loss: float
    task_type: str  # 'action' or 'severity'

@dataclass
class RebalancingConfig:
    """Configuration for rebalancing strategies"""
    # Performance thresholds
    min_class_recall: float = 0.3
    target_macro_recall: float = 0.7
    performance_window: int = 3  # epochs to consider for trend analysis
    
    # Adaptation parameters
    adaptation_rate: float = 0.1
    max_weight_multiplier: float = 10.0
    min_weight_multiplier: float = 0.1
    
    # Strategy selection
    primary_strategies: List[RebalancingStrategy] = None
    fallback_strategies: List[RebalancingStrategy] = None
    
    def __post_init__(self):
        if self.primary_strategies is None:
            self.primary_strategies = [
                RebalancingStrategy.WEIGHTED_SAMPLING,
                RebalancingStrategy.FOCAL_LOSS,
                RebalancingStrategy.CLASS_WEIGHTS
            ]
        if self.fallback_strategies is None:
            self.fallback_strategies = [
                RebalancingStrategy.OVERSAMPLING,
                RebalancingStrategy.MIXUP
            ]

class SmartRebalancer:
    """
    Adaptive rebalancing system that monitors performance and adjusts strategies
    """
    
    def __init__(self, 
                 num_action_classes: int = 8,
                 num_severity_classes: int = 4,
                 config: Optional[RebalancingConfig] = None,
                 save_dir: str = "rebalancing_logs"):
        
        self.num_action_classes = num_action_classes
        self.num_severity_classes = num_severity_classes
        self.config = config or RebalancingConfig()
        self.save_dir = save_dir
        
        # Performance tracking
        self.action_metrics_history: List[PerformanceMetrics] = []
        self.severity_metrics_history: List[PerformanceMetrics] = []
        
        # Current rebalancing parameters
        self.current_action_weights = torch.ones(num_action_classes)
        self.current_severity_weights = torch.ones(num_severity_classes)
        self.current_sampling_weights = None
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness = defaultdict(list)
        
        # Class distribution info
        self.class_distributions = {}
        
        # Setup logging
        self._setup_logging()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info("SmartRebalancer initialized")
    
    def _setup_logging(self):
        """Setup logging for the rebalancer"""
        self.logger = logging.getLogger('SmartRebalancer')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def initialize_class_distributions(self, 
                                     action_labels: List[torch.Tensor],
                                     severity_labels: List[torch.Tensor]):
        """Initialize class distribution information"""
        
        # Convert one-hot to class indices
        action_indices = torch.cat(action_labels, dim=0).argmax(dim=1).cpu().numpy()
        severity_indices = torch.cat(severity_labels, dim=0).argmax(dim=1).cpu().numpy()
        
        # Calculate distributions
        action_counts = Counter(action_indices)
        severity_counts = Counter(severity_indices)
        
        total_samples = len(action_indices)
        
        self.class_distributions = {
            'action': {
                'counts': action_counts,
                'frequencies': {i: action_counts.get(i, 0) / total_samples 
                              for i in range(self.num_action_classes)},
                'total_samples': total_samples
            },
            'severity': {
                'counts': severity_counts,
                'frequencies': {i: severity_counts.get(i, 0) / total_samples 
                              for i in range(self.num_severity_classes)},
                'total_samples': total_samples
            }
        }
        
        # Initialize weights based on inverse frequency
        self._initialize_weights()
        
        self.logger.info("Class distributions initialized")
        self.logger.info(f"Action frequencies: {self.class_distributions['action']['frequencies']}")
        self.logger.info(f"Severity frequencies: {self.class_distributions['severity']['frequencies']}")
    
    def _initialize_weights(self):
        """Initialize class weights based on inverse frequency"""
        
        # Action weights
        action_freqs = self.class_distributions['action']['frequencies']
        for i in range(self.num_action_classes):
            freq = action_freqs[i]
            if freq > 0:
                if freq > 0.4:  # Very dominant classes
                    self.current_action_weights[i] = max(0.9, 1.0 / (freq ** 0.2))
                elif freq > 0.25:  # Major classes
                    self.current_action_weights[i] = max(1.0, min(1.3, 1.0 / (freq ** 0.4)))
                elif freq > 0.1:  # Medium classes
                    self.current_action_weights[i] = min(1.6, 1.0 / (freq ** 0.5))
                elif freq > 0.03:  # Small classes
                    self.current_action_weights[i] = min(2.0, 1.0 / (freq ** 0.6))
                else:  # Very small classes
                    self.current_action_weights[i] = min(2.5, max(1.8, 1.0 / (freq ** 0.7)))
            else:
                self.current_action_weights[i] = 1.0
        
        # Severity weights (more aggressive)
        severity_freqs = self.class_distributions['severity']['frequencies']
        for i in range(self.num_severity_classes):
            freq = severity_freqs[i]
            if freq > 0:
                if freq > 0.5:  # Dominant class
                    self.current_severity_weights[i] = max(0.7, 1.0 / (freq ** 0.3))
                elif freq > 0.25:  # Major class
                    self.current_severity_weights[i] = min(2.0, 1.0 / (freq ** 0.6))
                elif freq > 0.05:  # Medium class
                    self.current_severity_weights[i] = min(3.5, 1.0 / (freq ** 0.8))
                else:  # Minority class (Red Card)
                    self.current_severity_weights[i] = min(10.0, max(5.0, 1.0 / (freq ** 0.9)))
            else:
                self.current_severity_weights[i] = 1.0
    
    def update_performance(self, 
                          epoch: int,
                          action_metrics: Dict[str, Any],
                          severity_metrics: Dict[str, Any]):
        """Update performance metrics and adapt strategies"""
        
        # Create performance metric objects
        action_perf = PerformanceMetrics(
            epoch=epoch,
            overall_accuracy=action_metrics['accuracy'],
            macro_f1=action_metrics['macro_f1'],
            macro_recall=action_metrics['macro_recall'],
            per_class_recall=action_metrics['per_class_recall'],
            per_class_precision=action_metrics['per_class_precision'],
            per_class_f1=action_metrics['per_class_f1'],
            loss=action_metrics['loss'],
            task_type='action'
        )
        
        severity_perf = PerformanceMetrics(
            epoch=epoch,
            overall_accuracy=severity_metrics['accuracy'],
            macro_f1=severity_metrics['macro_f1'],
            macro_recall=severity_metrics['macro_recall'],
            per_class_recall=severity_metrics['per_class_recall'],
            per_class_precision=severity_metrics['per_class_precision'],
            per_class_f1=severity_metrics['per_class_f1'],
            loss=severity_metrics['loss'],
            task_type='severity'
        )
        
        # Store metrics
        self.action_metrics_history.append(action_perf)
        self.severity_metrics_history.append(severity_perf)
        
        # Adapt strategies based on performance
        self._adapt_strategies(action_perf, severity_perf)
        
        # Save state
        self._save_state(epoch)
        
        self.logger.info(f"Epoch {epoch}: Updated performance metrics and adapted strategies")
    
    def _adapt_strategies(self, 
                         action_perf: PerformanceMetrics,
                         severity_perf: PerformanceMetrics):
        """Adapt rebalancing strategies based on performance"""
        
        # Analyze performance trends
        action_trend = self._analyze_performance_trend('action')
        severity_trend = self._analyze_performance_trend('severity')
        
        # Adapt action weights
        self._adapt_class_weights(action_perf, 'action', action_trend)
        
        # Adapt severity weights (more aggressive)
        self._adapt_class_weights(severity_perf, 'severity', severity_trend)
        
        # Update sampling weights
        self._update_sampling_weights()
    
    def _analyze_performance_trend(self, task_type: str) -> Dict[str, Any]:
        """Analyze performance trends for a task"""
        
        history = (self.action_metrics_history if task_type == 'action' 
                  else self.severity_metrics_history)
        
        if len(history) < 1:
            return {
                'trend': 'insufficient_data',
                'recall_trend': 'insufficient_data',
                'f1_trend': 'insufficient_data',
                'loss_trend': 'insufficient_data',
                'problematic_classes': [],
                'current_macro_recall': 0.0,
                'current_macro_f1': 0.0
            }
        
        # Look at recent performance window
        recent_window = min(self.config.performance_window, len(history))
        recent_metrics = history[-recent_window:]
        
        # Calculate trends
        macro_recalls = [m.macro_recall for m in recent_metrics]
        macro_f1s = [m.macro_f1 for m in recent_metrics]
        losses = [m.loss for m in recent_metrics]
        
        # Simple trend analysis
        if len(recent_metrics) >= 2:
            recall_trend = 'improving' if macro_recalls[-1] > macro_recalls[0] else 'declining'
            f1_trend = 'improving' if macro_f1s[-1] > macro_f1s[0] else 'declining'
            loss_trend = 'improving' if losses[-1] < losses[0] else 'declining'
        else:
            recall_trend = 'stable'
            f1_trend = 'stable'
            loss_trend = 'stable'
        
        # Identify problematic classes
        latest_metrics = recent_metrics[-1]
        problematic_classes = [
            class_id for class_id, recall in latest_metrics.per_class_recall.items()
            if recall < self.config.min_class_recall
        ]
        
        return {
            'trend': 'improving' if recall_trend == 'improving' and f1_trend == 'improving' else 'declining',
            'recall_trend': recall_trend,
            'f1_trend': f1_trend,
            'loss_trend': loss_trend,
            'problematic_classes': problematic_classes,
            'current_macro_recall': latest_metrics.macro_recall,
            'current_macro_f1': latest_metrics.macro_f1
        }
    
    def _adapt_class_weights(self, 
                           performance: PerformanceMetrics,
                           task_type: str,
                           trend_analysis: Dict[str, Any]):
        """Adapt class weights based on performance"""
        
        current_weights = (self.current_action_weights if task_type == 'action' 
                          else self.current_severity_weights)
        
        # Identify classes that need adjustment
        problematic_classes = trend_analysis['problematic_classes']
        
        for class_id in problematic_classes:
            current_recall = performance.per_class_recall.get(class_id, 0.0)
            
            if current_recall < self.config.min_class_recall:
                # Increase weight for underperforming classes
                adjustment_factor = 1.0 + self.config.adaptation_rate * (
                    self.config.min_class_recall - current_recall
                )
                
                new_weight = current_weights[class_id] * adjustment_factor
                new_weight = min(new_weight, self.config.max_weight_multiplier)
                
                current_weights[class_id] = new_weight
                
                self.logger.info(
                    f"Increased {task_type} class {class_id} weight to {new_weight:.3f} "
                    f"(recall: {current_recall:.3f})"
                )
        
        # Reduce weights for overperforming classes if overall performance is good
        if performance.macro_recall > self.config.target_macro_recall:
            for class_id, recall in performance.per_class_recall.items():
                if recall > 0.9:  # Very high recall
                    adjustment_factor = 1.0 - self.config.adaptation_rate * 0.5
                    new_weight = current_weights[class_id] * adjustment_factor
                    new_weight = max(new_weight, self.config.min_weight_multiplier)
                    
                    current_weights[class_id] = new_weight
    
    def _update_sampling_weights(self):
        """Update sampling weights based on current severity performance"""
        
        if not self.severity_metrics_history:
            return
        
        latest_severity = self.severity_metrics_history[-1]
        severity_counts = self.class_distributions['severity']['counts']
        total_samples = self.class_distributions['severity']['total_samples']
        
        # Create aggressive sampling weights based on severity performance
        sample_weights = []
        
        for i in range(total_samples):
            # This is a simplified approach - in practice, you'd need to map
            # sample indices to their severity classes
            # For now, we'll create weights based on class distribution
            pass
        
        # Generate sampling weights based on severity class frequencies and performance
        severity_freqs = self.class_distributions['severity']['frequencies']
        severity_performance = latest_severity.per_class_recall
        
        sampling_multipliers = {}
        for class_id in range(self.num_severity_classes):
            freq = severity_freqs[class_id]
            recall = severity_performance.get(class_id, 0.0)
            
            # Base weight from frequency
            if freq > 0.5:  # Dominant class
                base_weight = max(0.5, 1.0 / (freq ** 0.4))
            elif freq > 0.25:  # Major class
                base_weight = min(1.5, 1.0 / (freq ** 0.6))
            elif freq > 0.05:  # Medium class
                base_weight = min(3.0, 1.0 / (freq ** 0.8))
            else:  # Minority class (Red Card)
                base_weight = min(8.0, max(4.0, 1.0 / (freq ** 0.95)))
            
            # Adjust based on performance
            if recall < self.config.min_class_recall:
                performance_multiplier = 1.0 + (self.config.min_class_recall - recall)
            else:
                performance_multiplier = 1.0
            
            sampling_multipliers[class_id] = base_weight * performance_multiplier
        
        self.current_sampling_multipliers = sampling_multipliers
        
        self.logger.info(f"Updated sampling multipliers: {sampling_multipliers}")
    
    def get_class_weights(self, task_type: str, device: torch.device) -> torch.Tensor:
        """Get current class weights for loss functions"""
        
        if task_type == 'action':
            return self.current_action_weights.to(device)
        elif task_type == 'severity':
            return self.current_severity_weights.to(device)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_focal_loss_params(self, task_type: str) -> Dict[str, Any]:
        """Get adaptive focal loss parameters"""
        
        if not hasattr(self, f'{task_type}_metrics_history'):
            # Default parameters
            return {
                'gamma': 2.0 if task_type == 'severity' else 1.5,
                'alpha': None,
                'label_smoothing': 0.1 if task_type == 'severity' else 0.05
            }
        
        history = getattr(self, f'{task_type}_metrics_history')
        
        if not history:
            return {
                'gamma': 2.0 if task_type == 'severity' else 1.5,
                'alpha': None,
                'label_smoothing': 0.1 if task_type == 'severity' else 0.05
            }
        
        latest_performance = history[-1]
        
        # Adaptive gamma based on performance
        if latest_performance.macro_recall < 0.5:
            gamma = 3.0 if task_type == 'severity' else 2.0
        elif latest_performance.macro_recall < 0.7:
            gamma = 2.5 if task_type == 'severity' else 1.8
        else:
            gamma = 2.0 if task_type == 'severity' else 1.5
        
        # Adaptive label smoothing
        if latest_performance.macro_recall > 0.8:
            label_smoothing = 0.15 if task_type == 'severity' else 0.08
        else:
            label_smoothing = 0.1 if task_type == 'severity' else 0.05
        
        return {
            'gamma': gamma,
            'alpha': None,  # Will use class weights instead
            'label_smoothing': label_smoothing
        }
    
    def should_use_mixup(self, epoch: int) -> Tuple[bool, float]:
        """Determine if mixup should be used and with what alpha"""
        
        if not self.severity_metrics_history:
            return True, 0.2  # Default
        
        latest_severity = self.severity_metrics_history[-1]
        
        # Use mixup more aggressively for underperforming minority classes
        min_recall = min(latest_severity.per_class_recall.values())
        
        if min_recall < 0.3:
            return True, 0.4  # Strong mixup
        elif min_recall < 0.5:
            return True, 0.3  # Moderate mixup
        elif min_recall < 0.7:
            return True, 0.2  # Light mixup
        else:
            return False, 0.0  # No mixup needed
    
    def get_sampling_strategy_params(self) -> Dict[str, Any]:
        """Get parameters for sampling strategy"""
        
        if not hasattr(self, 'current_sampling_multipliers'):
            return {'strategy': 'weighted', 'multipliers': {}}
        
        return {
            'strategy': 'weighted',
            'multipliers': self.current_sampling_multipliers
        }
    
    def _save_state(self, epoch: int):
        """Save current state to disk"""
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'action_weights': self.current_action_weights.tolist(),
            'severity_weights': self.current_severity_weights.tolist(),
            'class_distributions': self.class_distributions,
            'config': {
                'min_class_recall': self.config.min_class_recall,
                'target_macro_recall': self.config.target_macro_recall,
                'adaptation_rate': self.config.adaptation_rate
            }
        }
        
        # Save performance history (last 10 epochs to avoid huge files)
        if len(self.action_metrics_history) > 10:
            recent_action = self.action_metrics_history[-10:]
        else:
            recent_action = self.action_metrics_history
            
        if len(self.severity_metrics_history) > 10:
            recent_severity = self.severity_metrics_history[-10:]
        else:
            recent_severity = self.severity_metrics_history
        
        state['recent_action_metrics'] = [
            {
                'epoch': int(m.epoch),
                'macro_recall': float(m.macro_recall),
                'macro_f1': float(m.macro_f1),
                'per_class_recall': {str(k): float(v) for k, v in m.per_class_recall.items()}
            } for m in recent_action
        ]
        
        state['recent_severity_metrics'] = [
            {
                'epoch': int(m.epoch),
                'macro_recall': float(m.macro_recall),
                'macro_f1': float(m.macro_f1),
                'per_class_recall': {str(k): float(v) for k, v in m.per_class_recall.items()}
            } for m in recent_severity
        ]
        
        save_path = os.path.join(self.save_dir, f'rebalancer_state_epoch_{epoch}.json')
        try:
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")
    
    def load_state(self, epoch: int) -> bool:
        """Load state from disk"""
        
        load_path = os.path.join(self.save_dir, f'rebalancer_state_epoch_{epoch}.json')
        
        if not os.path.exists(load_path):
            return False
        
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            self.current_action_weights = torch.tensor(state['action_weights'])
            self.current_severity_weights = torch.tensor(state['severity_weights'])
            self.class_distributions = state['class_distributions']
            
            self.logger.info(f"Loaded rebalancer state from epoch {epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """Get current strategy recommendations"""
        
        recommendations = {
            'use_focal_loss': True,
            'focal_loss_params': {
                'action': self.get_focal_loss_params('action'),
                'severity': self.get_focal_loss_params('severity')
            },
            'class_weights': {
                'action': self.current_action_weights.tolist(),
                'severity': self.current_severity_weights.tolist()
            },
            'sampling_strategy': self.get_sampling_strategy_params()
        }
        
        # Add mixup recommendation
        use_mixup, mixup_alpha = self.should_use_mixup(
            len(self.severity_metrics_history) if self.severity_metrics_history else 0
        )
        recommendations['mixup'] = {
            'use_mixup': use_mixup,
            'alpha': mixup_alpha
        }
        
        return recommendations