"""
Automated Hyperparameter Optimization System for MVFouls Performance Optimization

This module implements Bayesian optimization for key hyperparameters, performance-based
hyperparameter adjustment during training, sensitivity analysis, and automatic tuning
integrated with performance monitoring.

Requirements addressed:
- 4.3: Performance-based hyperparameter adjustment during training
- 4.4: Automatic hyperparameter tuning and sensitivity analysis
"""

import torch
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from enum import Enum
import pickle
import warnings

# Bayesian optimization imports
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Bayesian optimization will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    BAYESIAN_GP = "bayesian_gp"
    BAYESIAN_RF = "bayesian_rf"
    BAYESIAN_GBRT = "bayesian_gbrt"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    ADAPTIVE_ONLY = "adaptive_only"


class HyperparameterType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass
class HyperparameterSpace:
    """Definition of a hyperparameter search space"""
    name: str
    param_type: HyperparameterType
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[List[Any]] = None
    default: Any = None
    description: str = ""
    
    def to_skopt_dimension(self):
        """Convert to scikit-optimize dimension"""
        if not SKOPT_AVAILABLE:
            return None
            
        if self.param_type == HyperparameterType.CONTINUOUS:
            return Real(self.low, self.high, name=self.name)
        elif self.param_type == HyperparameterType.INTEGER:
            return Integer(int(self.low), int(self.high), name=self.name)
        elif self.param_type == HyperparameterType.CATEGORICAL:
            return Categorical(self.categories, name=self.name)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveAdjustment:
    """Record of an adaptive hyperparameter adjustment"""
    epoch: int
    parameter: str
    old_value: Any
    new_value: Any
    trigger_reason: str
    performance_metrics: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'parameter': self.parameter,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'trigger_reason': self.trigger_reason,
            'performance_metrics': self.performance_metrics,
            'timestamp': self.timestamp.isoformat()
        }


class BayesianHyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameter tuning using Gaussian Processes,
    Random Forests, or Gradient Boosted Trees.
    """
    
    def __init__(self,
                 parameter_space: List[HyperparameterSpace],
                 strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_GP,
                 n_calls: int = 50,
                 n_initial_points: int = 10,
                 acquisition_func: str = 'EI',
                 random_state: int = 42):
        """
        Initialize Bayesian hyperparameter optimizer.
        
        Args:
            parameter_space: List of hyperparameter definitions
            strategy: Optimization strategy to use
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            acquisition_func: Acquisition function ('EI', 'PI', 'LCB')
            random_state: Random state for reproducibility
        """
        self.parameter_space = parameter_space
        self.strategy = strategy
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_func = acquisition_func
        self.random_state = random_state
        
        # Convert to scikit-optimize dimensions
        if SKOPT_AVAILABLE:
            self.dimensions = [param.to_skopt_dimension() for param in parameter_space]
            self.param_names = [param.name for param in parameter_space]
        else:
            self.dimensions = None
            self.param_names = [param.name for param in parameter_space]
        
        # Optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        
        logger.info(f"BayesianHyperparameterOptimizer initialized:")
        logger.info(f"  Strategy: {strategy.value}")
        logger.info(f"  Parameters: {self.param_names}")
        logger.info(f"  Max evaluations: {n_calls}")
    
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Run Bayesian optimization to find best hyperparameters.
        
        Args:
            objective_function: Function to optimize (should return score to maximize)
            initial_params: Optional initial parameter values
            
        Returns:
            OptimizationResult with best parameters and optimization info
        """
        if not SKOPT_AVAILABLE and self.strategy.value.startswith('bayesian'):
            logger.warning("scikit-optimize not available. Falling back to random search.")
            return self._random_search_optimize(objective_function, initial_params)
        
        start_time = time.time()
        
        # Create objective function wrapper
        @use_named_args(self.dimensions)
        def objective(**params):
            try:
                # Convert parameters to proper types
                converted_params = self._convert_params(params)
                
                # Evaluate objective function
                score = objective_function(converted_params)
                
                # Record evaluation
                evaluation = {
                    'params': converted_params.copy(),
                    'score': score,
                    'evaluation_time': time.time(),
                    'evaluation_id': len(self.optimization_history)
                }
                self.optimization_history.append(evaluation)
                
                # Update best if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = converted_params.copy()
                    logger.info(f"New best score: {score:.4f} with params: {converted_params}")
                
                # Return negative score for minimization
                return -score
                
            except Exception as e:
                logger.error(f"Error in objective function evaluation: {e}")
                return 1e6  # Large penalty for failed evaluations
        
        # Add initial point if provided
        x0 = None
        if initial_params:
            x0 = [initial_params.get(name, self._get_default_value(name)) 
                  for name in self.param_names]
        
        # Run optimization based on strategy
        try:
            if self.strategy == OptimizationStrategy.BAYESIAN_GP:
                result = gp_minimize(
                    func=objective,
                    dimensions=self.dimensions,
                    n_calls=self.n_calls,
                    n_initial_points=self.n_initial_points,
                    acquisition_func=self.acquisition_func,
                    random_state=self.random_state,
                    x0=x0
                )
            elif self.strategy == OptimizationStrategy.BAYESIAN_RF:
                result = forest_minimize(
                    func=objective,
                    dimensions=self.dimensions,
                    n_calls=self.n_calls,
                    n_initial_points=self.n_initial_points,
                    acquisition_func=self.acquisition_func,
                    random_state=self.random_state,
                    x0=x0
                )
            elif self.strategy == OptimizationStrategy.BAYESIAN_GBRT:
                result = gbrt_minimize(
                    func=objective,
                    dimensions=self.dimensions,
                    n_calls=self.n_calls,
                    n_initial_points=self.n_initial_points,
                    acquisition_func=self.acquisition_func,
                    random_state=self.random_state,
                    x0=x0
                )
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")
            
            optimization_time = time.time() - start_time
            
            # Extract convergence information
            convergence_info = {
                'final_score': -result.fun,
                'n_evaluations': len(result.func_vals),
                'convergence_delta': abs(result.func_vals[-1] - result.func_vals[-min(5, len(result.func_vals))]),
                'acquisition_function': self.acquisition_func
            }
            
            return OptimizationResult(
                best_params=self.best_params,
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                total_evaluations=len(self.optimization_history),
                optimization_time=optimization_time,
                convergence_info=convergence_info
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._fallback_optimization(objective_function, initial_params)
    
    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameters to proper types based on parameter space"""
        converted = {}
        for param_def in self.parameter_space:
            if param_def.name in params:
                value = params[param_def.name]
                if param_def.param_type == HyperparameterType.INTEGER:
                    converted[param_def.name] = int(value)
                else:
                    converted[param_def.name] = value
            else:
                converted[param_def.name] = param_def.default
        return converted
    
    def _get_default_value(self, param_name: str) -> Any:
        """Get default value for a parameter"""
        for param_def in self.parameter_space:
            if param_def.name == param_name:
                return param_def.default
        return None
    
    def _random_search_optimize(self, 
                               objective_function: Callable,
                               initial_params: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Fallback random search optimization"""
        logger.info("Running random search optimization...")
        
        start_time = time.time()
        
        # Evaluate initial params if provided
        if initial_params:
            try:
                score = objective_function(initial_params)
                self.optimization_history.append({
                    'params': initial_params.copy(),
                    'score': score,
                    'evaluation_time': time.time(),
                    'evaluation_id': 0
                })
                self.best_score = score
                self.best_params = initial_params.copy()
            except Exception as e:
                logger.error(f"Failed to evaluate initial params: {e}")
        
        # Random search
        for i in range(self.n_calls - (1 if initial_params else 0)):
            # Generate random parameters
            params = {}
            for param_def in self.parameter_space:
                if param_def.param_type == HyperparameterType.CONTINUOUS:
                    params[param_def.name] = np.random.uniform(param_def.low, param_def.high)
                elif param_def.param_type == HyperparameterType.INTEGER:
                    params[param_def.name] = np.random.randint(int(param_def.low), int(param_def.high) + 1)
                elif param_def.param_type == HyperparameterType.CATEGORICAL:
                    params[param_def.name] = np.random.choice(param_def.categories)
            
            try:
                score = objective_function(params)
                
                evaluation = {
                    'params': params.copy(),
                    'score': score,
                    'evaluation_time': time.time(),
                    'evaluation_id': len(self.optimization_history)
                }
                self.optimization_history.append(evaluation)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    logger.info(f"New best score: {score:.4f} with params: {params}")
                    
            except Exception as e:
                logger.error(f"Error in random search evaluation {i}: {e}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            optimization_history=self.optimization_history,
            total_evaluations=len(self.optimization_history),
            optimization_time=optimization_time,
            convergence_info={'method': 'random_search'}
        )
    
    def _fallback_optimization(self, 
                              objective_function: Callable,
                              initial_params: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Fallback optimization when Bayesian methods fail"""
        logger.warning("Bayesian optimization failed. Using random search fallback.")
        return self._random_search_optimize(objective_function, initial_params)


class AdaptiveHyperparameterAdjuster:
    """
    Adaptive hyperparameter adjustment during training based on performance metrics.
    """
    
    def __init__(self,
                 adjustment_rules: Dict[str, Dict] = None,
                 adaptation_frequency: int = 5,
                 min_epochs_between_adjustments: int = 3,
                 performance_window: int = 5):
        """
        Initialize adaptive hyperparameter adjuster.
        
        Args:
            adjustment_rules: Rules for parameter adjustments
            adaptation_frequency: How often to check for adjustments (epochs)
            min_epochs_between_adjustments: Minimum epochs between adjustments
            performance_window: Window size for performance trend analysis
        """
        self.adjustment_rules = adjustment_rules or self._get_default_adjustment_rules()
        self.adaptation_frequency = adaptation_frequency
        self.min_epochs_between_adjustments = min_epochs_between_adjustments
        self.performance_window = performance_window
        
        # Tracking
        self.adjustment_history = []
        self.last_adjustment_epoch = defaultdict(lambda: -1)
        self.performance_history = deque(maxlen=performance_window)
        
        logger.info("AdaptiveHyperparameterAdjuster initialized")
        logger.info(f"  Adjustment frequency: {adaptation_frequency} epochs")
        logger.info(f"  Rules: {list(self.adjustment_rules.keys())}")
    
    def _get_default_adjustment_rules(self) -> Dict[str, Dict]:
        """Get default adjustment rules for common hyperparameters"""
        return {
            'learning_rate': {
                'triggers': [
                    {
                        'condition': 'plateau',
                        'metric': 'combined_macro_recall',
                        'threshold': 0.01,
                        'window': 3,
                        'action': 'multiply',
                        'factor': 0.5,
                        'min_value': 1e-6
                    },
                    {
                        'condition': 'performance_drop',
                        'metric': 'combined_macro_recall',
                        'threshold': 0.05,
                        'action': 'multiply',
                        'factor': 0.3,
                        'min_value': 1e-6
                    }
                ]
            },
            'focal_loss_gamma': {
                'triggers': [
                    {
                        'condition': 'minority_class_zero_recall',
                        'classes': [4, 7, 3],  # Pushing, Dive, Red Card
                        'action': 'increase',
                        'factor': 1.2,
                        'max_value': 5.0
                    },
                    {
                        'condition': 'minority_class_improved',
                        'classes': [4, 7, 3],
                        'threshold': 0.1,
                        'action': 'decrease',
                        'factor': 0.9,
                        'min_value': 0.5
                    }
                ]
            },
            'class_weights': {
                'triggers': [
                    {
                        'condition': 'class_recall_below_threshold',
                        'threshold': 0.05,
                        'action': 'multiply',
                        'factor': 1.5,
                        'max_value': 10.0
                    }
                ]
            },
            'gradient_accumulation_steps': {
                'triggers': [
                    {
                        'condition': 'gradient_norm_high',
                        'threshold': 10.0,
                        'action': 'increase',
                        'factor': 1.5,
                        'max_value': 16
                    },
                    {
                        'condition': 'gradient_norm_low',
                        'threshold': 0.1,
                        'action': 'decrease',
                        'factor': 0.8,
                        'min_value': 1
                    }
                ]
            }
        }
    
    def check_and_adjust(self,
                        epoch: int,
                        performance_metrics: Dict[str, Any],
                        current_params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[AdaptiveAdjustment]]:
        """
        Check if hyperparameters should be adjusted and return new values.
        
        Args:
            epoch: Current epoch
            performance_metrics: Current performance metrics
            current_params: Current hyperparameter values
            
        Returns:
            Tuple of (updated_params, list_of_adjustments)
        """
        # Update performance history
        self.performance_history.append({
            'epoch': epoch,
            'metrics': performance_metrics.copy()
        })
        
        # Check if it's time to evaluate adjustments
        if epoch % self.adaptation_frequency != 0:
            return current_params, []
        
        adjustments = []
        updated_params = current_params.copy()
        
        # Check each parameter for adjustment triggers
        for param_name, rules in self.adjustment_rules.items():
            if param_name not in current_params:
                continue
            
            # Check if enough time has passed since last adjustment
            if epoch - self.last_adjustment_epoch[param_name] < self.min_epochs_between_adjustments:
                continue
            
            # Check each trigger rule
            for rule in rules.get('triggers', []):
                adjustment = self._check_trigger(
                    param_name, rule, epoch, performance_metrics, current_params[param_name]
                )
                
                if adjustment:
                    adjustments.append(adjustment)
                    updated_params[param_name] = adjustment.new_value
                    self.last_adjustment_epoch[param_name] = epoch
                    self.adjustment_history.append(adjustment)
                    
                    logger.info(f"Adaptive adjustment at epoch {epoch}:")
                    logger.info(f"  {param_name}: {adjustment.old_value} -> {adjustment.new_value}")
                    logger.info(f"  Reason: {adjustment.trigger_reason}")
                    break  # Only one adjustment per parameter per check
        
        return updated_params, adjustments
    
    def _check_trigger(self,
                      param_name: str,
                      rule: Dict,
                      epoch: int,
                      metrics: Dict[str, Any],
                      current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check if a specific trigger condition is met"""
        condition = rule['condition']
        
        if condition == 'plateau':
            return self._check_plateau_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'performance_drop':
            return self._check_performance_drop_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'minority_class_zero_recall':
            return self._check_minority_zero_recall_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'minority_class_improved':
            return self._check_minority_improved_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'class_recall_below_threshold':
            return self._check_class_recall_threshold_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'gradient_norm_high':
            return self._check_gradient_norm_high_trigger(param_name, rule, epoch, metrics, current_value)
        elif condition == 'gradient_norm_low':
            return self._check_gradient_norm_low_trigger(param_name, rule, epoch, metrics, current_value)
        
        return None
    
    def _check_plateau_trigger(self, param_name: str, rule: Dict, epoch: int, 
                              metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for performance plateau"""
        metric_name = rule['metric']
        threshold = rule['threshold']
        window = rule.get('window', 3)
        
        if len(self.performance_history) < window:
            return None
        
        # Get recent metric values
        recent_values = [h['metrics'].get(metric_name, 0) for h in list(self.performance_history)[-window:]]
        
        if len(recent_values) < window:
            return None
        
        # Check if improvement is below threshold
        improvement = max(recent_values) - min(recent_values)
        if improvement < threshold:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Plateau detected: {metric_name} improvement {improvement:.4f} < {threshold}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_performance_drop_trigger(self, param_name: str, rule: Dict, epoch: int,
                                       metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for performance drop"""
        metric_name = rule['metric']
        threshold = rule['threshold']
        
        if len(self.performance_history) < 2:
            return None
        
        current_metric = metrics.get(metric_name, 0)
        previous_metric = self.performance_history[-2]['metrics'].get(metric_name, 0)
        
        drop = previous_metric - current_metric
        if drop > threshold:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Performance drop: {metric_name} dropped by {drop:.4f}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_minority_zero_recall_trigger(self, param_name: str, rule: Dict, epoch: int,
                                           metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for minority classes with zero recall"""
        classes = rule.get('classes', [])
        
        # Check action classes
        action_recalls = metrics.get('action_class_recall', [])
        severity_recalls = metrics.get('severity_class_recall', [])
        
        zero_recall_classes = []
        
        # Check action minority classes (4=Pushing, 7=Dive)
        for class_id in [4, 7]:
            if class_id in classes and class_id < len(action_recalls):
                if action_recalls[class_id] == 0.0:
                    zero_recall_classes.append(f"action_{class_id}")
        
        # Check severity minority class (3=Red Card)
        if 3 in classes and 3 < len(severity_recalls):
            if severity_recalls[3] == 0.0:
                zero_recall_classes.append("severity_3")
        
        if zero_recall_classes:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Zero recall for minority classes: {zero_recall_classes}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_minority_improved_trigger(self, param_name: str, rule: Dict, epoch: int,
                                        metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check if minority classes have improved above threshold"""
        classes = rule.get('classes', [])
        threshold = rule.get('threshold', 0.1)
        
        action_recalls = metrics.get('action_class_recall', [])
        severity_recalls = metrics.get('severity_class_recall', [])
        
        improved_classes = []
        
        # Check action minority classes
        for class_id in [4, 7]:
            if class_id in classes and class_id < len(action_recalls):
                if action_recalls[class_id] >= threshold:
                    improved_classes.append(f"action_{class_id}")
        
        # Check severity minority class
        if 3 in classes and 3 < len(severity_recalls):
            if severity_recalls[3] >= threshold:
                improved_classes.append("severity_3")
        
        if improved_classes:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Minority classes improved above {threshold}: {improved_classes}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_class_recall_threshold_trigger(self, param_name: str, rule: Dict, epoch: int,
                                             metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for classes below recall threshold"""
        threshold = rule.get('threshold', 0.05)
        
        action_recalls = metrics.get('action_class_recall', [])
        severity_recalls = metrics.get('severity_class_recall', [])
        
        below_threshold = []
        
        for i, recall in enumerate(action_recalls):
            if recall < threshold:
                below_threshold.append(f"action_{i}")
        
        for i, recall in enumerate(severity_recalls):
            if recall < threshold:
                below_threshold.append(f"severity_{i}")
        
        if below_threshold:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Classes below recall threshold {threshold}: {below_threshold}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_gradient_norm_high_trigger(self, param_name: str, rule: Dict, epoch: int,
                                         metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for high gradient norm"""
        threshold = rule.get('threshold', 10.0)
        gradient_norm = metrics.get('gradient_norm', 0.0)
        
        if gradient_norm > threshold:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"High gradient norm: {gradient_norm:.2f} > {threshold}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_gradient_norm_low_trigger(self, param_name: str, rule: Dict, epoch: int,
                                        metrics: Dict, current_value: Any) -> Optional[AdaptiveAdjustment]:
        """Check for low gradient norm"""
        threshold = rule.get('threshold', 0.1)
        gradient_norm = metrics.get('gradient_norm', 1.0)
        
        if gradient_norm < threshold:
            new_value = self._apply_adjustment_action(current_value, rule)
            
            return AdaptiveAdjustment(
                epoch=epoch,
                parameter=param_name,
                old_value=current_value,
                new_value=new_value,
                trigger_reason=f"Low gradient norm: {gradient_norm:.4f} < {threshold}",
                performance_metrics=metrics.copy(),
                timestamp=datetime.now()
            )
        
        return None
    
    def _apply_adjustment_action(self, current_value: Any, rule: Dict) -> Any:
        """Apply adjustment action to current value"""
        action = rule['action']
        
        if action == 'multiply':
            factor = rule.get('factor', 1.0)
            new_value = current_value * factor
        elif action == 'increase':
            factor = rule.get('factor', 1.1)
            new_value = current_value * factor
        elif action == 'decrease':
            factor = rule.get('factor', 0.9)
            new_value = current_value * factor
        elif action == 'add':
            delta = rule.get('delta', 0.1)
            new_value = current_value + delta
        elif action == 'subtract':
            delta = rule.get('delta', 0.1)
            new_value = current_value - delta
        else:
            new_value = current_value
        
        # Apply bounds
        if 'min_value' in rule:
            new_value = max(new_value, rule['min_value'])
        if 'max_value' in rule:
            new_value = min(new_value, rule['max_value'])
        
        return new_value
    
    def get_adjustment_history(self) -> List[Dict]:
        """Get history of all adjustments"""
        return [adj.to_dict() for adj in self.adjustment_history]


class HyperparameterSensitivityAnalyzer:
    """
    Analyze hyperparameter sensitivity and importance.
    """
    
    def __init__(self,
                 parameter_space: List[HyperparameterSpace],
                 n_samples: int = 100,
                 analysis_methods: List[str] = None):
        """
        Initialize sensitivity analyzer.
        
        Args:
            parameter_space: List of hyperparameter definitions
            n_samples: Number of samples for sensitivity analysis
            analysis_methods: Methods to use for analysis
        """
        self.parameter_space = parameter_space
        self.n_samples = n_samples
        self.analysis_methods = analysis_methods or ['sobol', 'morris', 'correlation']
        
        self.sensitivity_results = {}
        self.parameter_importance = {}
        
        logger.info(f"HyperparameterSensitivityAnalyzer initialized:")
        logger.info(f"  Parameters: {[p.name for p in parameter_space]}")
        logger.info(f"  Analysis methods: {self.analysis_methods}")
    
    def analyze_sensitivity(self,
                           evaluation_function: Callable,
                           baseline_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on hyperparameters.
        
        Args:
            evaluation_function: Function to evaluate parameter combinations
            baseline_params: Baseline parameter values
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info("Starting hyperparameter sensitivity analysis...")
        
        results = {}
        
        # One-at-a-time sensitivity analysis
        results['one_at_a_time'] = self._one_at_a_time_analysis(
            evaluation_function, baseline_params
        )
        
        # Correlation analysis
        if 'correlation' in self.analysis_methods:
            results['correlation'] = self._correlation_analysis(
                evaluation_function, baseline_params
            )
        
        # Morris method (if available)
        if 'morris' in self.analysis_methods:
            results['morris'] = self._morris_analysis(
                evaluation_function, baseline_params
            )
        
        # Calculate parameter importance ranking
        results['importance_ranking'] = self._calculate_importance_ranking(results)
        
        self.sensitivity_results = results
        
        logger.info("Sensitivity analysis completed")
        return results
    
    def _one_at_a_time_analysis(self,
                               evaluation_function: Callable,
                               baseline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one-at-a-time sensitivity analysis"""
        logger.info("Running one-at-a-time sensitivity analysis...")
        
        baseline_score = evaluation_function(baseline_params)
        sensitivities = {}
        
        for param_def in self.parameter_space:
            param_name = param_def.name
            param_sensitivities = []
            
            if param_def.param_type == HyperparameterType.CONTINUOUS:
                # Test multiple values across the range
                test_values = np.linspace(param_def.low, param_def.high, 5)
            elif param_def.param_type == HyperparameterType.INTEGER:
                # Test integer values across the range
                test_values = np.linspace(int(param_def.low), int(param_def.high), 
                                        min(5, int(param_def.high) - int(param_def.low) + 1))
                test_values = [int(v) for v in test_values]
            elif param_def.param_type == HyperparameterType.CATEGORICAL:
                test_values = param_def.categories
            else:
                continue
            
            for test_value in test_values:
                if test_value == baseline_params.get(param_name):
                    continue
                
                test_params = baseline_params.copy()
                test_params[param_name] = test_value
                
                try:
                    test_score = evaluation_function(test_params)
                    sensitivity = abs(test_score - baseline_score)
                    param_sensitivities.append({
                        'value': test_value,
                        'score': test_score,
                        'sensitivity': sensitivity
                    })
                except Exception as e:
                    logger.warning(f"Failed to evaluate {param_name}={test_value}: {e}")
            
            if param_sensitivities:
                sensitivities[param_name] = {
                    'mean_sensitivity': np.mean([s['sensitivity'] for s in param_sensitivities]),
                    'max_sensitivity': max([s['sensitivity'] for s in param_sensitivities]),
                    'evaluations': param_sensitivities
                }
        
        return sensitivities
    
    def _correlation_analysis(self,
                             evaluation_function: Callable,
                             baseline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation-based sensitivity analysis"""
        logger.info("Running correlation sensitivity analysis...")
        
        # Generate random parameter combinations
        param_combinations = []
        scores = []
        
        for _ in range(self.n_samples):
            params = baseline_params.copy()
            
            # Randomly vary each parameter
            for param_def in self.parameter_space:
                if param_def.param_type == HyperparameterType.CONTINUOUS:
                    params[param_def.name] = np.random.uniform(param_def.low, param_def.high)
                elif param_def.param_type == HyperparameterType.INTEGER:
                    params[param_def.name] = np.random.randint(int(param_def.low), int(param_def.high) + 1)
                elif param_def.param_type == HyperparameterType.CATEGORICAL:
                    params[param_def.name] = np.random.choice(param_def.categories)
            
            try:
                score = evaluation_function(params)
                param_combinations.append(params)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to evaluate parameter combination: {e}")
        
        # Calculate correlations
        correlations = {}
        if len(scores) > 10:  # Need sufficient samples
            for param_def in self.parameter_space:
                param_name = param_def.name
                
                if param_def.param_type in [HyperparameterType.CONTINUOUS, HyperparameterType.INTEGER]:
                    param_values = [combo[param_name] for combo in param_combinations]
                    correlation = np.corrcoef(param_values, scores)[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations[param_name] = {
                            'correlation': correlation,
                            'abs_correlation': abs(correlation),
                            'samples': len(param_values)
                        }
        
        return correlations
    
    def _morris_analysis(self,
                        evaluation_function: Callable,
                        baseline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Morris method sensitivity analysis (simplified version)"""
        logger.info("Running Morris method sensitivity analysis...")
        
        # Simplified Morris method implementation
        morris_results = {}
        
        for param_def in self.parameter_space:
            param_name = param_def.name
            
            if param_def.param_type not in [HyperparameterType.CONTINUOUS, HyperparameterType.INTEGER]:
                continue
            
            elementary_effects = []
            
            # Generate trajectories
            for _ in range(min(20, self.n_samples // len(self.parameter_space))):
                # Create base point
                base_params = baseline_params.copy()
                
                # Randomly vary other parameters
                for other_param in self.parameter_space:
                    if other_param.name != param_name:
                        if other_param.param_type == HyperparameterType.CONTINUOUS:
                            base_params[other_param.name] = np.random.uniform(
                                other_param.low, other_param.high
                            )
                        elif other_param.param_type == HyperparameterType.INTEGER:
                            base_params[other_param.name] = np.random.randint(
                                int(other_param.low), int(other_param.high) + 1
                            )
                
                # Calculate elementary effect
                try:
                    base_score = evaluation_function(base_params)
                    
                    # Perturb target parameter
                    perturbed_params = base_params.copy()
                    if param_def.param_type == HyperparameterType.CONTINUOUS:
                        delta = (param_def.high - param_def.low) * 0.1
                        perturbed_params[param_name] = min(
                            param_def.high, base_params[param_name] + delta
                        )
                    elif param_def.param_type == HyperparameterType.INTEGER:
                        perturbed_params[param_name] = min(
                            int(param_def.high), base_params[param_name] + 1
                        )
                    
                    perturbed_score = evaluation_function(perturbed_params)
                    elementary_effect = perturbed_score - base_score
                    elementary_effects.append(elementary_effect)
                    
                except Exception as e:
                    logger.warning(f"Failed Morris evaluation for {param_name}: {e}")
            
            if elementary_effects:
                morris_results[param_name] = {
                    'mu': np.mean(elementary_effects),
                    'mu_star': np.mean(np.abs(elementary_effects)),
                    'sigma': np.std(elementary_effects),
                    'n_effects': len(elementary_effects)
                }
        
        return morris_results
    
    def _calculate_importance_ranking(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate parameter importance ranking from analysis results"""
        importance_scores = defaultdict(list)
        
        # Collect importance scores from different methods
        if 'one_at_a_time' in results:
            for param_name, data in results['one_at_a_time'].items():
                importance_scores[param_name].append(data['mean_sensitivity'])
        
        if 'correlation' in results:
            for param_name, data in results['correlation'].items():
                importance_scores[param_name].append(data['abs_correlation'])
        
        if 'morris' in results:
            for param_name, data in results['morris'].items():
                importance_scores[param_name].append(data['mu_star'])
        
        # Calculate combined importance score
        combined_importance = []
        for param_name, scores in importance_scores.items():
            if scores:
                combined_score = np.mean(scores)
                combined_importance.append({
                    'parameter': param_name,
                    'importance_score': combined_score,
                    'individual_scores': scores,
                    'rank': 0  # Will be filled after sorting
                })
        
        # Sort by importance and assign ranks
        combined_importance.sort(key=lambda x: x['importance_score'], reverse=True)
        for i, item in enumerate(combined_importance):
            item['rank'] = i + 1
        
        return combined_importance
    
    def get_most_important_parameters(self, top_k: int = 5) -> List[str]:
        """Get the most important parameters"""
        if 'importance_ranking' in self.sensitivity_results:
            ranking = self.sensitivity_results['importance_ranking']
            return [item['parameter'] for item in ranking[:top_k]]
        return []


class AutomatedHyperparameterOptimizer:
    """
    Main class that coordinates Bayesian optimization, adaptive adjustment,
    and sensitivity analysis for automated hyperparameter optimization.
    """
    
    def __init__(self,
                 parameter_space: List[HyperparameterSpace],
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_GP,
                 save_dir: str = "hyperparameter_optimization",
                 enable_adaptive_adjustment: bool = True,
                 enable_sensitivity_analysis: bool = True):
        """
        Initialize automated hyperparameter optimizer.
        
        Args:
            parameter_space: List of hyperparameter definitions
            optimization_strategy: Strategy for Bayesian optimization
            save_dir: Directory to save optimization results
            enable_adaptive_adjustment: Whether to enable adaptive adjustments
            enable_sensitivity_analysis: Whether to enable sensitivity analysis
        """
        self.parameter_space = parameter_space
        self.optimization_strategy = optimization_strategy
        self.save_dir = save_dir
        self.enable_adaptive_adjustment = enable_adaptive_adjustment
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.bayesian_optimizer = BayesianHyperparameterOptimizer(
            parameter_space=parameter_space,
            strategy=optimization_strategy
        )
        
        if enable_adaptive_adjustment:
            self.adaptive_adjuster = AdaptiveHyperparameterAdjuster()
        else:
            self.adaptive_adjuster = None
        
        if enable_sensitivity_analysis:
            self.sensitivity_analyzer = HyperparameterSensitivityAnalyzer(
                parameter_space=parameter_space
            )
        else:
            self.sensitivity_analyzer = None
        
        # Optimization state
        self.optimization_results = None
        self.current_best_params = None
        self.optimization_history = []
        
        logger.info("AutomatedHyperparameterOptimizer initialized successfully!")
        logger.info(f"  Parameters: {[p.name for p in parameter_space]}")
        logger.info(f"  Strategy: {optimization_strategy.value}")
        logger.info(f"  Adaptive adjustment: {enable_adaptive_adjustment}")
        logger.info(f"  Sensitivity analysis: {enable_sensitivity_analysis}")
    
    def optimize_hyperparameters(self,
                                objective_function: Callable,
                                initial_params: Optional[Dict[str, Any]] = None,
                                n_calls: int = 50) -> OptimizationResult:
        """
        Run Bayesian optimization to find optimal hyperparameters.
        
        Args:
            objective_function: Function to optimize (returns score to maximize)
            initial_params: Optional initial parameter values
            n_calls: Number of optimization calls
            
        Returns:
            OptimizationResult with best parameters and optimization info
        """
        logger.info("Starting Bayesian hyperparameter optimization...")
        
        # Update optimizer settings
        self.bayesian_optimizer.n_calls = n_calls
        
        # Run optimization
        self.optimization_results = self.bayesian_optimizer.optimize(
            objective_function=objective_function,
            initial_params=initial_params
        )
        
        self.current_best_params = self.optimization_results.best_params
        
        # Save results
        self._save_optimization_results()
        
        logger.info(f"Optimization completed. Best score: {self.optimization_results.best_score:.4f}")
        logger.info(f"Best parameters: {self.current_best_params}")
        
        return self.optimization_results
    
    def adaptive_adjust_during_training(self,
                                      epoch: int,
                                      performance_metrics: Dict[str, Any],
                                      current_params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[AdaptiveAdjustment]]:
        """
        Perform adaptive hyperparameter adjustment during training.
        
        Args:
            epoch: Current training epoch
            performance_metrics: Current performance metrics
            current_params: Current hyperparameter values
            
        Returns:
            Tuple of (updated_params, list_of_adjustments)
        """
        if not self.adaptive_adjuster:
            return current_params, []
        
        updated_params, adjustments = self.adaptive_adjuster.check_and_adjust(
            epoch=epoch,
            performance_metrics=performance_metrics,
            current_params=current_params
        )
        
        # Save adjustment history
        if adjustments:
            self._save_adaptive_adjustments(adjustments)
        
        return updated_params, adjustments
    
    def analyze_parameter_sensitivity(self,
                                    evaluation_function: Callable,
                                    baseline_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on hyperparameters.
        
        Args:
            evaluation_function: Function to evaluate parameter combinations
            baseline_params: Baseline parameter values
            
        Returns:
            Sensitivity analysis results
        """
        if not self.sensitivity_analyzer:
            logger.warning("Sensitivity analysis is disabled")
            return {}
        
        logger.info("Starting hyperparameter sensitivity analysis...")
        
        results = self.sensitivity_analyzer.analyze_sensitivity(
            evaluation_function=evaluation_function,
            baseline_params=baseline_params
        )
        
        # Save results
        self._save_sensitivity_results(results)
        
        logger.info("Sensitivity analysis completed")
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        summary = {
            'optimization_completed': self.optimization_results is not None,
            'current_best_params': self.current_best_params,
            'parameter_space': [p.name for p in self.parameter_space],
            'optimization_strategy': self.optimization_strategy.value,
            'adaptive_adjustment_enabled': self.adaptive_adjuster is not None,
            'sensitivity_analysis_enabled': self.sensitivity_analyzer is not None
        }
        
        if self.optimization_results:
            summary.update({
                'best_score': self.optimization_results.best_score,
                'total_evaluations': self.optimization_results.total_evaluations,
                'optimization_time': self.optimization_results.optimization_time
            })
        
        if self.adaptive_adjuster:
            summary['adaptive_adjustments'] = len(self.adaptive_adjuster.adjustment_history)
        
        if self.sensitivity_analyzer and self.sensitivity_analyzer.sensitivity_results:
            summary['most_important_parameters'] = self.sensitivity_analyzer.get_most_important_parameters()
        
        return summary
    
    def _save_optimization_results(self):
        """Save Bayesian optimization results"""
        if not self.optimization_results:
            return
        
        results_file = os.path.join(self.save_dir, "bayesian_optimization_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.optimization_results.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {results_file}")
    
    def _save_adaptive_adjustments(self, adjustments: List[AdaptiveAdjustment]):
        """Save adaptive adjustment history"""
        adjustments_file = os.path.join(self.save_dir, "adaptive_adjustments.json")
        
        # Load existing adjustments
        existing_adjustments = []
        if os.path.exists(adjustments_file):
            try:
                with open(adjustments_file, 'r') as f:
                    existing_adjustments = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing adjustments: {e}")
        
        # Add new adjustments
        new_adjustments = [adj.to_dict() for adj in adjustments]
        existing_adjustments.extend(new_adjustments)
        
        # Save updated adjustments
        with open(adjustments_file, 'w') as f:
            json.dump(existing_adjustments, f, indent=2, default=str)
        
        logger.info(f"Adaptive adjustments saved to {adjustments_file}")
    
    def _save_sensitivity_results(self, results: Dict[str, Any]):
        """Save sensitivity analysis results"""
        sensitivity_file = os.path.join(self.save_dir, "sensitivity_analysis.json")
        
        with open(sensitivity_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Sensitivity analysis results saved to {sensitivity_file}")


def create_mvfouls_parameter_space() -> List[HyperparameterSpace]:
    """
    Create parameter space for MVFouls optimization.
    
    Returns:
        List of hyperparameter definitions for MVFouls training
    """
    return [
        HyperparameterSpace(
            name="learning_rate",
            param_type=HyperparameterType.CONTINUOUS,
            low=1e-6,
            high=1e-2,
            default=1e-4,
            description="Learning rate for optimizer"
        ),
        HyperparameterSpace(
            name="focal_loss_gamma",
            param_type=HyperparameterType.CONTINUOUS,
            low=0.5,
            high=5.0,
            default=2.0,
            description="Gamma parameter for focal loss"
        ),
        HyperparameterSpace(
            name="batch_size",
            param_type=HyperparameterType.INTEGER,
            low=2,
            high=16,
            default=4,
            description="Training batch size"
        ),
        HyperparameterSpace(
            name="gradient_accumulation_steps",
            param_type=HyperparameterType.INTEGER,
            low=1,
            high=16,
            default=4,
            description="Number of gradient accumulation steps"
        ),
        HyperparameterSpace(
            name="weight_decay",
            param_type=HyperparameterType.CONTINUOUS,
            low=1e-6,
            high=1e-2,
            default=1e-4,
            description="Weight decay for regularization"
        ),
        HyperparameterSpace(
            name="dropout_rate",
            param_type=HyperparameterType.CONTINUOUS,
            low=0.0,
            high=0.5,
            default=0.1,
            description="Dropout rate for regularization"
        ),
        HyperparameterSpace(
            name="label_smoothing",
            param_type=HyperparameterType.CONTINUOUS,
            low=0.0,
            high=0.2,
            default=0.0,
            description="Label smoothing factor"
        ),
        HyperparameterSpace(
            name="scheduler_T_0",
            param_type=HyperparameterType.INTEGER,
            low=5,
            high=20,
            default=10,
            description="Initial restart period for cosine annealing"
        ),
        HyperparameterSpace(
            name="scheduler_T_mult",
            param_type=HyperparameterType.INTEGER,
            low=1,
            high=3,
            default=2,
            description="Restart period multiplier for cosine annealing"
        ),
        HyperparameterSpace(
            name="mixup_alpha",
            param_type=HyperparameterType.CONTINUOUS,
            low=0.0,
            high=1.0,
            default=0.2,
            description="Alpha parameter for mixup augmentation"
        )
    ]


if __name__ == "__main__":
    print("Testing Automated Hyperparameter Optimization System...")
    
    # Create parameter space
    parameter_space = create_mvfouls_parameter_space()
    print(f"✓ Created parameter space with {len(parameter_space)} parameters")
    
    # Initialize optimizer
    optimizer = AutomatedHyperparameterOptimizer(
        parameter_space=parameter_space,
        optimization_strategy=OptimizationStrategy.BAYESIAN_GP,
        enable_adaptive_adjustment=True,
        enable_sensitivity_analysis=True
    )
    print("✓ AutomatedHyperparameterOptimizer initialized")
    
    # Test objective function
    def dummy_objective(params):
        # Simulate training and return combined macro recall
        lr = params.get('learning_rate', 1e-4)
        gamma = params.get('focal_loss_gamma', 2.0)
        batch_size = params.get('batch_size', 4)
        
        # Simulate some realistic behavior
        score = 0.4 + 0.1 * np.log10(lr / 1e-4) + 0.05 * (gamma - 2.0) - 0.01 * abs(batch_size - 4)
        score += np.random.normal(0, 0.02)  # Add noise
        return max(0.0, min(1.0, score))
    
    # Test Bayesian optimization (small number of calls for testing)
    try:
        result = optimizer.optimize_hyperparameters(
            objective_function=dummy_objective,
            n_calls=10
        )
        print(f"✓ Bayesian optimization completed")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best params: {result.best_params}")
    except Exception as e:
        print(f"✗ Bayesian optimization failed: {e}")
    
    # Test adaptive adjustment
    try:
        current_params = {
            'learning_rate': 1e-4,
            'focal_loss_gamma': 2.0,
            'batch_size': 4
        }
        
        performance_metrics = {
            'combined_macro_recall': 0.35,
            'action_class_recall': [0.8, 0.7, 0.6, 0.5, 0.0, 0.4, 0.3, 0.0],  # Pushing=0, Dive=0
            'severity_class_recall': [0.6, 0.8, 0.7, 0.0],  # Red Card=0
            'gradient_norm': 5.0
        }
        
        updated_params, adjustments = optimizer.adaptive_adjust_during_training(
            epoch=10,
            performance_metrics=performance_metrics,
            current_params=current_params
        )
        
        print(f"✓ Adaptive adjustment completed")
        print(f"  Adjustments made: {len(adjustments)}")
        if adjustments:
            for adj in adjustments:
                print(f"    {adj.parameter}: {adj.old_value} -> {adj.new_value}")
    except Exception as e:
        print(f"✗ Adaptive adjustment failed: {e}")
    
    # Test sensitivity analysis (small sample for testing)
    try:
        baseline_params = {
            'learning_rate': 1e-4,
            'focal_loss_gamma': 2.0,
            'batch_size': 4,
            'gradient_accumulation_steps': 4
        }
        
        # Use smaller sample size for testing
        optimizer.sensitivity_analyzer.n_samples = 20
        
        sensitivity_results = optimizer.analyze_parameter_sensitivity(
            evaluation_function=dummy_objective,
            baseline_params=baseline_params
        )
        
        print(f"✓ Sensitivity analysis completed")
        if 'importance_ranking' in sensitivity_results:
            print("  Parameter importance ranking:")
            for item in sensitivity_results['importance_ranking'][:3]:
                print(f"    {item['rank']}. {item['parameter']}: {item['importance_score']:.4f}")
    except Exception as e:
        print(f"✗ Sensitivity analysis failed: {e}")
    
    # Test summary
    try:
        summary = optimizer.get_optimization_summary()
        print(f"✓ Optimization summary generated")
        print(f"  Optimization completed: {summary['optimization_completed']}")
        print(f"  Adaptive adjustments: {summary.get('adaptive_adjustments', 0)}")
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
    
    print("\nAutomated Hyperparameter Optimization System test completed!")