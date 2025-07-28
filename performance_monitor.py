"""
Comprehensive Performance Monitoring and Alert System for MVFouls Training

This module provides real-time per-class metrics tracking, alert system for performance issues,
detailed logging with confusion matrices, and automatic corrective action triggers.

Requirements addressed:
- 8.1: Real-time per-class metrics tracking
- 8.2: Alert system for zero recall and performance drops
- 8.3: Detailed logging with confusion matrices and trend analysis
- 8.4: Automatic corrective action triggers
"""

import torch
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging
from enum import Enum


class AlertType(Enum):
    ZERO_RECALL = "zero_recall"
    PERFORMANCE_DROP = "performance_drop"
    PLATEAU = "plateau"
    GRADIENT_EXPLOSION = "gradient_explosion"
    LOSS_DIVERGENCE = "loss_divergence"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClassMetrics:
    """Metrics for a single class"""
    class_id: int
    class_name: str
    recall: float
    precision: float
    f1_score: float
    support: int
    trend: str  # 'improving', 'declining', 'stable'
    epochs_since_improvement: int
    best_recall: float
    best_precision: float
    best_f1: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceAlert:
    """Alert for performance issues"""
    alert_type: AlertType
    severity: AlertSeverity
    affected_classes: List[int]
    message: str
    suggested_actions: List[str]
    timestamp: datetime
    epoch: int
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'affected_classes': self.affected_classes,
            'message': self.message,
            'suggested_actions': self.suggested_actions,
            'timestamp': self.timestamp.isoformat(),
            'epoch': self.epoch,
            'metrics': self.metrics
        }


@dataclass
class EpochMetrics:
    """Complete metrics for an epoch"""
    epoch: int
    timestamp: datetime
    action_metrics: Dict[int, ClassMetrics]
    severity_metrics: Dict[int, ClassMetrics]
    macro_recall_action: float
    macro_recall_severity: float
    combined_macro_recall: float
    loss_action: float
    loss_severity: float
    total_loss: float
    learning_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'timestamp': self.timestamp.isoformat(),
            'action_metrics': {k: v.to_dict() for k, v in self.action_metrics.items()},
            'severity_metrics': {k: v.to_dict() for k, v in self.severity_metrics.items()},
            'macro_recall_action': self.macro_recall_action,
            'macro_recall_severity': self.macro_recall_severity,
            'combined_macro_recall': self.combined_macro_recall,
            'loss_action': self.loss_action,
            'loss_severity': self.loss_severity,
            'total_loss': self.total_loss,
            'learning_rate': self.learning_rate
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for MVFouls training
    
    Features:
    - Real-time per-class metrics tracking
    - Performance trend analysis
    - Alert system for performance issues
    - Detailed logging and visualization
    - Automatic corrective action suggestions
    """
    
    def __init__(self, 
                 action_class_names: Dict[int, str],
                 severity_class_names: Dict[int, str],
                 log_dir: str = "performance_logs",
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 trend_window: int = 5):
        """
        Initialize the performance monitor
        
        Args:
            action_class_names: Mapping of action class IDs to names
            severity_class_names: Mapping of severity class IDs to names
            log_dir: Directory to save logs and visualizations
            alert_thresholds: Custom thresholds for alerts
            trend_window: Number of epochs to consider for trend analysis
        """
        self.action_class_names = action_class_names
        self.severity_class_names = severity_class_names
        self.log_dir = log_dir
        self.trend_window = trend_window
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'zero_recall_epochs': 3,  # Alert if recall is 0 for this many epochs
            'performance_drop_threshold': 0.1,  # Alert if recall drops by this much
            'plateau_epochs': 5,  # Alert if no improvement for this many epochs
            'min_acceptable_recall': 0.05,  # Minimum acceptable recall
            'target_combined_recall': 0.45,  # Target combined macro recall
            'gradient_norm_threshold': 10.0,  # Gradient explosion threshold
            'loss_divergence_threshold': 2.0  # Loss divergence multiplier
        }
        
        # Storage for metrics history
        self.epoch_metrics_history: List[EpochMetrics] = []
        self.alerts_history: List[PerformanceAlert] = []
        
        # Per-class tracking
        self.action_recall_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trend_window))
        self.severity_recall_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trend_window))
        
        # Best metrics tracking
        self.best_action_metrics: Dict[int, ClassMetrics] = {}
        self.best_severity_metrics: Dict[int, ClassMetrics] = {}
        self.best_combined_recall = 0.0
        
        # Current epoch tracking
        self.current_epoch = 0
        self.zero_recall_counters: Dict[str, Dict[int, int]] = {
            'action': defaultdict(int),
            'severity': defaultdict(int)
        }
        
        self.logger.info("PerformanceMonitor initialized successfully")
        self.logger.info(f"Monitoring {len(action_class_names)} action classes and {len(severity_class_names)} severity classes")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.log_dir, f"performance_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PerformanceMonitor')
    
    def update_metrics(self, 
                      epoch: int,
                      action_predictions: torch.Tensor,
                      action_targets: torch.Tensor,
                      severity_predictions: torch.Tensor,
                      severity_targets: torch.Tensor,
                      loss_action: float,
                      loss_severity: float,
                      learning_rate: float,
                      gradient_norm: Optional[float] = None) -> EpochMetrics:
        """
        Update metrics for the current epoch
        
        Args:
            epoch: Current epoch number
            action_predictions: Action predictions tensor
            action_targets: Action ground truth tensor
            severity_predictions: Severity predictions tensor
            severity_targets: Severity ground truth tensor
            loss_action: Action loss value
            loss_severity: Severity loss value
            learning_rate: Current learning rate
            gradient_norm: Gradient norm for monitoring
            
        Returns:
            EpochMetrics object with computed metrics
        """
        self.current_epoch = epoch
        
        # Convert predictions to class indices
        action_pred_classes = torch.argmax(action_predictions, dim=1).cpu().numpy()
        action_true_classes = torch.argmax(action_targets, dim=1).cpu().numpy()
        severity_pred_classes = torch.argmax(severity_predictions, dim=1).cpu().numpy()
        severity_true_classes = torch.argmax(severity_targets, dim=1).cpu().numpy()
        
        # Calculate per-class metrics
        action_metrics = self._calculate_class_metrics(
            action_true_classes, action_pred_classes, 
            self.action_class_names, 'action'
        )
        
        severity_metrics = self._calculate_class_metrics(
            severity_true_classes, severity_pred_classes,
            self.severity_class_names, 'severity'
        )
        
        # Calculate macro recalls
        macro_recall_action = np.mean([m.recall for m in action_metrics.values()])
        macro_recall_severity = np.mean([m.recall for m in severity_metrics.values()])
        combined_macro_recall = (macro_recall_action + macro_recall_severity) / 2
        
        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            timestamp=datetime.now(),
            action_metrics=action_metrics,
            severity_metrics=severity_metrics,
            macro_recall_action=macro_recall_action,
            macro_recall_severity=macro_recall_severity,
            combined_macro_recall=combined_macro_recall,
            loss_action=loss_action,
            loss_severity=loss_severity,
            total_loss=loss_action + loss_severity,
            learning_rate=learning_rate
        )
        
        # Update history
        self.epoch_metrics_history.append(epoch_metrics)
        
        # Update recall history for trend analysis
        for class_id, metrics in action_metrics.items():
            self.action_recall_history[class_id].append(metrics.recall)
        
        for class_id, metrics in severity_metrics.items():
            self.severity_recall_history[class_id].append(metrics.recall)
        
        # Update best metrics
        self._update_best_metrics(action_metrics, severity_metrics, combined_macro_recall)
        
        # Check for alerts
        alerts = self._check_alerts(epoch_metrics, gradient_norm)
        
        # Log metrics
        self._log_epoch_metrics(epoch_metrics, alerts)
        
        # Save metrics to file
        self._save_metrics(epoch_metrics)
        
        return epoch_metrics
    
    def _calculate_class_metrics(self, 
                                true_labels: np.ndarray,
                                pred_labels: np.ndarray,
                                class_names: Dict[int, str],
                                task_type: str) -> Dict[int, ClassMetrics]:
        """Calculate per-class metrics"""
        metrics = {}
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        for class_id, class_name in class_names.items():
            if class_id < len(precision):  # Ensure class exists in predictions
                # Determine trend
                trend = self._calculate_trend(class_id, recall[class_id], task_type)
                
                # Count epochs since improvement
                epochs_since_improvement = self._count_epochs_since_improvement(
                    class_id, recall[class_id], task_type
                )
                
                # Get best metrics
                best_recall = self._get_best_metric(class_id, 'recall', task_type)
                best_precision = self._get_best_metric(class_id, 'precision', task_type)
                best_f1 = self._get_best_metric(class_id, 'f1', task_type)
                
                metrics[class_id] = ClassMetrics(
                    class_id=class_id,
                    class_name=class_name,
                    recall=float(recall[class_id]),
                    precision=float(precision[class_id]),
                    f1_score=float(f1[class_id]),
                    support=int(support[class_id]),
                    trend=trend,
                    epochs_since_improvement=epochs_since_improvement,
                    best_recall=best_recall,
                    best_precision=best_precision,
                    best_f1=best_f1
                )
        
        return metrics
    
    def _calculate_trend(self, class_id: int, current_recall: float, task_type: str) -> str:
        """Calculate trend for a class based on recent recall history"""
        history = (self.action_recall_history[class_id] if task_type == 'action' 
                  else self.severity_recall_history[class_id])
        
        if len(history) < 2:
            return 'stable'
        
        recent_values = list(history)[-3:]  # Look at last 3 values
        
        if len(recent_values) >= 2:
            if recent_values[-1] > recent_values[-2] * 1.05:  # 5% improvement
                return 'improving'
            elif recent_values[-1] < recent_values[-2] * 0.95:  # 5% decline
                return 'declining'
        
        return 'stable'
    
    def _count_epochs_since_improvement(self, class_id: int, current_recall: float, task_type: str) -> int:
        """Count epochs since last improvement for a class"""
        history = (self.action_recall_history[class_id] if task_type == 'action' 
                  else self.severity_recall_history[class_id])
        
        if len(history) == 0:
            return 0
        
        max_recall = max(history)
        if current_recall >= max_recall * 0.99:  # Within 1% of best
            return 0
        
        # Count epochs since best performance
        epochs_since = 0
        for i in range(len(history) - 1, -1, -1):
            if history[i] >= max_recall * 0.99:
                break
            epochs_since += 1
        
        return epochs_since
    
    def _get_best_metric(self, class_id: int, metric_type: str, task_type: str) -> float:
        """Get best historical metric for a class"""
        best_metrics = (self.best_action_metrics if task_type == 'action' 
                       else self.best_severity_metrics)
        
        if class_id not in best_metrics:
            return 0.0
        
        return getattr(best_metrics[class_id], f'best_{metric_type}', 0.0)
    
    def _update_best_metrics(self, 
                           action_metrics: Dict[int, ClassMetrics],
                           severity_metrics: Dict[int, ClassMetrics],
                           combined_recall: float):
        """Update best metrics tracking"""
        # Update best action metrics
        for class_id, metrics in action_metrics.items():
            if (class_id not in self.best_action_metrics or 
                metrics.recall > self.best_action_metrics[class_id].recall):
                self.best_action_metrics[class_id] = metrics
        
        # Update best severity metrics
        for class_id, metrics in severity_metrics.items():
            if (class_id not in self.best_severity_metrics or 
                metrics.recall > self.best_severity_metrics[class_id].recall):
                self.best_severity_metrics[class_id] = metrics
        
        # Update best combined recall
        if combined_recall > self.best_combined_recall:
            self.best_combined_recall = combined_recall
    
    def _check_alerts(self, epoch_metrics: EpochMetrics, gradient_norm: Optional[float] = None) -> List[PerformanceAlert]:
        """Check for performance alerts"""
        alerts = []
        
        # Check for zero recall
        alerts.extend(self._check_zero_recall_alerts(epoch_metrics))
        
        # Check for performance drops
        alerts.extend(self._check_performance_drop_alerts(epoch_metrics))
        
        # Check for plateaus
        alerts.extend(self._check_plateau_alerts(epoch_metrics))
        
        # Check for gradient issues
        if gradient_norm is not None:
            alerts.extend(self._check_gradient_alerts(gradient_norm))
        
        # Check for loss divergence
        alerts.extend(self._check_loss_divergence_alerts(epoch_metrics))
        
        # Store alerts
        self.alerts_history.extend(alerts)
        
        return alerts
    
    def _check_zero_recall_alerts(self, epoch_metrics: EpochMetrics) -> List[PerformanceAlert]:
        """Check for zero recall alerts"""
        alerts = []
        
        # Check action classes
        for class_id, metrics in epoch_metrics.action_metrics.items():
            if metrics.recall == 0.0:
                self.zero_recall_counters['action'][class_id] += 1
                
                if self.zero_recall_counters['action'][class_id] >= self.alert_thresholds['zero_recall_epochs']:
                    alerts.append(PerformanceAlert(
                        alert_type=AlertType.ZERO_RECALL,
                        severity=AlertSeverity.HIGH,
                        affected_classes=[class_id],
                        message=f"Action class '{metrics.class_name}' has zero recall for {self.zero_recall_counters['action'][class_id]} consecutive epochs",
                        suggested_actions=[
                            f"Increase sampling weight for class {class_id}",
                            f"Generate synthetic samples for class {class_id}",
                            f"Increase loss weight for class {class_id}",
                            "Check data quality and labeling"
                        ],
                        timestamp=datetime.now(),
                        epoch=epoch_metrics.epoch,
                        metrics={'recall': metrics.recall, 'support': metrics.support}
                    ))
            else:
                self.zero_recall_counters['action'][class_id] = 0
        
        # Check severity classes
        for class_id, metrics in epoch_metrics.severity_metrics.items():
            if metrics.recall == 0.0:
                self.zero_recall_counters['severity'][class_id] += 1
                
                if self.zero_recall_counters['severity'][class_id] >= self.alert_thresholds['zero_recall_epochs']:
                    alerts.append(PerformanceAlert(
                        alert_type=AlertType.ZERO_RECALL,
                        severity=AlertSeverity.HIGH,
                        affected_classes=[class_id],
                        message=f"Severity class '{metrics.class_name}' has zero recall for {self.zero_recall_counters['severity'][class_id]} consecutive epochs",
                        suggested_actions=[
                            f"Increase sampling weight for class {class_id}",
                            f"Generate synthetic samples for class {class_id}",
                            f"Increase loss weight for class {class_id}",
                            "Check data quality and labeling"
                        ],
                        timestamp=datetime.now(),
                        epoch=epoch_metrics.epoch,
                        metrics={'recall': metrics.recall, 'support': metrics.support}
                    ))
            else:
                self.zero_recall_counters['severity'][class_id] = 0
        
        return alerts
    
    def _check_performance_drop_alerts(self, epoch_metrics: EpochMetrics) -> List[PerformanceAlert]:
        """Check for significant performance drops"""
        alerts = []
        
        if len(self.epoch_metrics_history) < 2:
            return alerts
        
        prev_metrics = self.epoch_metrics_history[-2]
        
        # Check action classes
        for class_id, current_metrics in epoch_metrics.action_metrics.items():
            if class_id in prev_metrics.action_metrics:
                prev_recall = prev_metrics.action_metrics[class_id].recall
                current_recall = current_metrics.recall
                
                if (prev_recall > 0 and 
                    current_recall < prev_recall - self.alert_thresholds['performance_drop_threshold']):
                    
                    alerts.append(PerformanceAlert(
                        alert_type=AlertType.PERFORMANCE_DROP,
                        severity=AlertSeverity.MEDIUM,
                        affected_classes=[class_id],
                        message=f"Action class '{current_metrics.class_name}' recall dropped from {prev_recall:.3f} to {current_recall:.3f}",
                        suggested_actions=[
                            "Check for overfitting",
                            "Reduce learning rate",
                            "Increase regularization",
                            "Review recent data changes"
                        ],
                        timestamp=datetime.now(),
                        epoch=epoch_metrics.epoch,
                        metrics={'prev_recall': prev_recall, 'current_recall': current_recall}
                    ))
        
        # Check severity classes
        for class_id, current_metrics in epoch_metrics.severity_metrics.items():
            if class_id in prev_metrics.severity_metrics:
                prev_recall = prev_metrics.severity_metrics[class_id].recall
                current_recall = current_metrics.recall
                
                if (prev_recall > 0 and 
                    current_recall < prev_recall - self.alert_thresholds['performance_drop_threshold']):
                    
                    alerts.append(PerformanceAlert(
                        alert_type=AlertType.PERFORMANCE_DROP,
                        severity=AlertSeverity.MEDIUM,
                        affected_classes=[class_id],
                        message=f"Severity class '{current_metrics.class_name}' recall dropped from {prev_recall:.3f} to {current_recall:.3f}",
                        suggested_actions=[
                            "Check for overfitting",
                            "Reduce learning rate",
                            "Increase regularization",
                            "Review recent data changes"
                        ],
                        timestamp=datetime.now(),
                        epoch=epoch_metrics.epoch,
                        metrics={'prev_recall': prev_recall, 'current_recall': current_recall}
                    ))
        
        return alerts
    
    def _check_plateau_alerts(self, epoch_metrics: EpochMetrics) -> List[PerformanceAlert]:
        """Check for performance plateaus"""
        alerts = []
        
        if len(self.epoch_metrics_history) < self.alert_thresholds['plateau_epochs']:
            return alerts
        
        # Check combined macro recall plateau
        recent_recalls = [m.combined_macro_recall for m in self.epoch_metrics_history[-self.alert_thresholds['plateau_epochs']:]]
        
        if len(set([round(r, 3) for r in recent_recalls])) == 1:  # All values are the same
            alerts.append(PerformanceAlert(
                alert_type=AlertType.PLATEAU,
                severity=AlertSeverity.MEDIUM,
                affected_classes=[],
                message=f"Combined macro recall has plateaued at {recent_recalls[-1]:.3f} for {self.alert_thresholds['plateau_epochs']} epochs",
                suggested_actions=[
                    "Adjust learning rate schedule",
                    "Increase data augmentation",
                    "Try different optimization strategy",
                    "Review model architecture"
                ],
                timestamp=datetime.now(),
                epoch=epoch_metrics.epoch,
                metrics={'plateau_value': recent_recalls[-1], 'epochs': self.alert_thresholds['plateau_epochs']}
            ))
        
        return alerts
    
    def _check_gradient_alerts(self, gradient_norm: float) -> List[PerformanceAlert]:
        """Check for gradient-related issues"""
        alerts = []
        
        if gradient_norm > self.alert_thresholds['gradient_norm_threshold']:
            alerts.append(PerformanceAlert(
                alert_type=AlertType.GRADIENT_EXPLOSION,
                severity=AlertSeverity.HIGH,
                affected_classes=[],
                message=f"Gradient norm is {gradient_norm:.2f}, exceeding threshold of {self.alert_thresholds['gradient_norm_threshold']}",
                suggested_actions=[
                    "Apply gradient clipping",
                    "Reduce learning rate",
                    "Check for numerical instability",
                    "Review loss function scaling"
                ],
                timestamp=datetime.now(),
                epoch=self.current_epoch,
                metrics={'gradient_norm': gradient_norm}
            ))
        
        return alerts
    
    def _check_loss_divergence_alerts(self, epoch_metrics: EpochMetrics) -> List[PerformanceAlert]:
        """Check for loss divergence"""
        alerts = []
        
        if len(self.epoch_metrics_history) < 3:
            return alerts
        
        recent_losses = [m.total_loss for m in self.epoch_metrics_history[-3:]]
        
        if recent_losses[-1] > recent_losses[0] * self.alert_thresholds['loss_divergence_threshold']:
            alerts.append(PerformanceAlert(
                alert_type=AlertType.LOSS_DIVERGENCE,
                severity=AlertSeverity.HIGH,
                affected_classes=[],
                message=f"Loss has increased from {recent_losses[0]:.4f} to {recent_losses[-1]:.4f} over 3 epochs",
                suggested_actions=[
                    "Reduce learning rate immediately",
                    "Revert to previous checkpoint",
                    "Check for data corruption",
                    "Review recent hyperparameter changes"
                ],
                timestamp=datetime.now(),
                epoch=epoch_metrics.epoch,
                metrics={'initial_loss': recent_losses[0], 'current_loss': recent_losses[-1]}
            ))
        
        return alerts
    
    def _log_epoch_metrics(self, epoch_metrics: EpochMetrics, alerts: List[PerformanceAlert]):
        """Log epoch metrics and alerts"""
        self.logger.info(f"=== Epoch {epoch_metrics.epoch} Performance Summary ===")
        self.logger.info(f"Combined Macro Recall: {epoch_metrics.combined_macro_recall:.4f}")
        self.logger.info(f"Action Macro Recall: {epoch_metrics.macro_recall_action:.4f}")
        self.logger.info(f"Severity Macro Recall: {epoch_metrics.macro_recall_severity:.4f}")
        self.logger.info(f"Total Loss: {epoch_metrics.total_loss:.4f}")
        
        # Log per-class metrics
        self.logger.info("Action Class Performance:")
        for class_id, metrics in epoch_metrics.action_metrics.items():
            self.logger.info(f"  {metrics.class_name}: Recall={metrics.recall:.3f}, "
                           f"Precision={metrics.precision:.3f}, F1={metrics.f1_score:.3f}, "
                           f"Trend={metrics.trend}")
        
        self.logger.info("Severity Class Performance:")
        for class_id, metrics in epoch_metrics.severity_metrics.items():
            self.logger.info(f"  {metrics.class_name}: Recall={metrics.recall:.3f}, "
                           f"Precision={metrics.precision:.3f}, F1={metrics.f1_score:.3f}, "
                           f"Trend={metrics.trend}")
        
        # Log alerts
        if alerts:
            self.logger.warning(f"Generated {len(alerts)} alerts:")
            for alert in alerts:
                self.logger.warning(f"  {alert.severity.value.upper()}: {alert.message}")
                for action in alert.suggested_actions:
                    self.logger.warning(f"    - {action}")
        
        self.logger.info("=" * 50)
    
    def _save_metrics(self, epoch_metrics: EpochMetrics):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.log_dir, f"epoch_{epoch_metrics.epoch:03d}_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(epoch_metrics.to_dict(), f, indent=2)
        
        # Save cumulative metrics
        cumulative_file = os.path.join(self.log_dir, "cumulative_metrics.json")
        cumulative_data = {
            'epochs': [m.to_dict() for m in self.epoch_metrics_history],
            'alerts': [a.to_dict() for a in self.alerts_history],
            'best_combined_recall': self.best_combined_recall
        }
        
        with open(cumulative_file, 'w') as f:
            json.dump(cumulative_data, f, indent=2)
    
    def get_current_metrics(self) -> Optional[EpochMetrics]:
        """Get the most recent epoch metrics"""
        return self.epoch_metrics_history[-1] if self.epoch_metrics_history else None
    
    def get_performance_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get performance alerts, optionally filtered by severity"""
        if severity_filter is None:
            return self.alerts_history.copy()
        
        return [alert for alert in self.alerts_history if alert.severity == severity_filter]
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.epoch_metrics_history:
            return {"error": "No metrics available"}
        
        latest_metrics = self.epoch_metrics_history[-1]
        
        report = {
            "summary": {
                "total_epochs": len(self.epoch_metrics_history),
                "current_combined_recall": latest_metrics.combined_macro_recall,
                "best_combined_recall": self.best_combined_recall,
                "target_recall": self.alert_thresholds['target_combined_recall'],
                "target_achieved": latest_metrics.combined_macro_recall >= self.alert_thresholds['target_combined_recall'],
                "total_alerts": len(self.alerts_history),
                "high_severity_alerts": len([a for a in self.alerts_history if a.severity == AlertSeverity.HIGH])
            },
            "action_classes": {},
            "severity_classes": {},
            "trends": self._analyze_trends(),
            "recommendations": self._generate_recommendations()
        }
        
        # Add per-class analysis
        for class_id, metrics in latest_metrics.action_metrics.items():
            report["action_classes"][class_id] = {
                "name": metrics.class_name,
                "current_recall": metrics.recall,
                "best_recall": metrics.best_recall,
                "trend": metrics.trend,
                "epochs_since_improvement": metrics.epochs_since_improvement,
                "meets_minimum": metrics.recall >= self.alert_thresholds['min_acceptable_recall']
            }
        
        for class_id, metrics in latest_metrics.severity_metrics.items():
            report["severity_classes"][class_id] = {
                "name": metrics.class_name,
                "current_recall": metrics.recall,
                "best_recall": metrics.best_recall,
                "trend": metrics.trend,
                "epochs_since_improvement": metrics.epochs_since_improvement,
                "meets_minimum": metrics.recall >= self.alert_thresholds['min_acceptable_recall']
            }
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        if len(self.epoch_metrics_history) < 3:
            return {"insufficient_data": True}
        
        # Analyze combined recall trend
        recent_recalls = [m.combined_macro_recall for m in self.epoch_metrics_history[-5:]]
        
        trend_analysis = {
            "combined_recall_trend": "stable",
            "improving_classes": [],
            "declining_classes": [],
            "stagnant_classes": []
        }
        
        # Determine overall trend
        if len(recent_recalls) >= 3:
            if recent_recalls[-1] > recent_recalls[0] * 1.05:
                trend_analysis["combined_recall_trend"] = "improving"
            elif recent_recalls[-1] < recent_recalls[0] * 0.95:
                trend_analysis["combined_recall_trend"] = "declining"
        
        # Analyze per-class trends
        latest_metrics = self.epoch_metrics_history[-1]
        
        for class_id, metrics in latest_metrics.action_metrics.items():
            if metrics.trend == "improving":
                trend_analysis["improving_classes"].append(f"Action: {metrics.class_name}")
            elif metrics.trend == "declining":
                trend_analysis["declining_classes"].append(f"Action: {metrics.class_name}")
            elif metrics.epochs_since_improvement > 5:
                trend_analysis["stagnant_classes"].append(f"Action: {metrics.class_name}")
        
        for class_id, metrics in latest_metrics.severity_metrics.items():
            if metrics.trend == "improving":
                trend_analysis["improving_classes"].append(f"Severity: {metrics.class_name}")
            elif metrics.trend == "declining":
                trend_analysis["declining_classes"].append(f"Severity: {metrics.class_name}")
            elif metrics.epochs_since_improvement > 5:
                trend_analysis["stagnant_classes"].append(f"Severity: {metrics.class_name}")
        
        return trend_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current performance"""
        recommendations = []
        
        if not self.epoch_metrics_history:
            return recommendations
        
        latest_metrics = self.epoch_metrics_history[-1]
        
        # Check if target is met
        if latest_metrics.combined_macro_recall < self.alert_thresholds['target_combined_recall']:
            recommendations.append(f"Combined macro recall ({latest_metrics.combined_macro_recall:.3f}) is below target ({self.alert_thresholds['target_combined_recall']:.3f})")
        
        # Check for zero recall classes
        zero_recall_classes = []
        for class_id, metrics in latest_metrics.action_metrics.items():
            if metrics.recall == 0.0:
                zero_recall_classes.append(f"Action: {metrics.class_name}")
        
        for class_id, metrics in latest_metrics.severity_metrics.items():
            if metrics.recall == 0.0:
                zero_recall_classes.append(f"Severity: {metrics.class_name}")
        
        if zero_recall_classes:
            recommendations.append(f"Classes with zero recall need immediate attention: {', '.join(zero_recall_classes)}")
        
        # Check for recent alerts
        recent_alerts = [a for a in self.alerts_history if a.epoch >= latest_metrics.epoch - 2]
        if recent_alerts:
            high_priority_alerts = [a for a in recent_alerts if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]
            if high_priority_alerts:
                recommendations.append(f"Address {len(high_priority_alerts)} high-priority alerts from recent epochs")
        
        # Performance-based recommendations
        if latest_metrics.combined_macro_recall < 0.3:
            recommendations.append("Consider major architectural changes or data augmentation strategies")
        elif latest_metrics.combined_macro_recall < 0.4:
            recommendations.append("Focus on minority class sampling and loss function tuning")
        
        return recommendations
    
    def create_confusion_matrix_plot(self, 
                                   true_labels: np.ndarray,
                                   pred_labels: np.ndarray,
                                   class_names: List[str],
                                   task_type: str,
                                   epoch: int,
                                   save_path: Optional[str] = None) -> str:
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{task_type.capitalize()} Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, f'confusion_matrix_{task_type}_epoch_{epoch:03d}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_performance_trend_plot(self, save_path: Optional[str] = None) -> str:
        """Create performance trend plot"""
        if len(self.epoch_metrics_history) < 2:
            return ""
        
        epochs = [m.epoch for m in self.epoch_metrics_history]
        combined_recalls = [m.combined_macro_recall for m in self.epoch_metrics_history]
        action_recalls = [m.macro_recall_action for m in self.epoch_metrics_history]
        severity_recalls = [m.macro_recall_severity for m in self.epoch_metrics_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, combined_recalls, 'b-', linewidth=2, label='Combined Macro Recall')
        plt.plot(epochs, action_recalls, 'r--', label='Action Macro Recall')
        plt.plot(epochs, severity_recalls, 'g--', label='Severity Macro Recall')
        plt.axhline(y=self.alert_thresholds['target_combined_recall'], color='orange', 
                   linestyle=':', label=f'Target ({self.alert_thresholds["target_combined_recall"]:.2f})')
        plt.xlabel('Epoch')
        plt.ylabel('Macro Recall')
        plt.title('Performance Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss trends
        total_losses = [m.total_loss for m in self.epoch_metrics_history]
        action_losses = [m.loss_action for m in self.epoch_metrics_history]
        severity_losses = [m.loss_severity for m in self.epoch_metrics_history]
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
        plt.plot(epochs, action_losses, 'r--', label='Action Loss')
        plt.plot(epochs, severity_losses, 'g--', label='Severity Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'performance_trends.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def trigger_corrective_actions(self, alerts: List[PerformanceAlert]) -> Dict[str, Any]:
        """
        Trigger automatic corrective actions based on alerts
        
        Returns:
            Dictionary with suggested parameter adjustments
        """
        corrections = {
            'sampling_weights': {},
            'loss_weights': {},
            'learning_rate_adjustment': 1.0,
            'early_stopping': False,
            'checkpoint_revert': False,
            'synthetic_data_generation': []
        }
        
        for alert in alerts:
            if alert.alert_type == AlertType.ZERO_RECALL:
                # Increase sampling and loss weights for zero recall classes
                for class_id in alert.affected_classes:
                    corrections['sampling_weights'][class_id] = 3.0  # Triple sampling weight
                    corrections['loss_weights'][class_id] = 2.0     # Double loss weight
                    corrections['synthetic_data_generation'].append(class_id)
            
            elif alert.alert_type == AlertType.PERFORMANCE_DROP:
                # Reduce learning rate
                corrections['learning_rate_adjustment'] *= 0.5
            
            elif alert.alert_type == AlertType.GRADIENT_EXPLOSION:
                # Significant learning rate reduction
                corrections['learning_rate_adjustment'] *= 0.1
                corrections['checkpoint_revert'] = True
            
            elif alert.alert_type == AlertType.LOSS_DIVERGENCE:
                # Revert to previous checkpoint
                corrections['checkpoint_revert'] = True
                corrections['learning_rate_adjustment'] *= 0.2
            
            elif alert.alert_type == AlertType.PLATEAU:
                # Trigger early stopping consideration
                corrections['early_stopping'] = True
        
        return corrections


# Example usage and integration functions
def create_performance_monitor(action_class_names: Dict[int, str],
                             severity_class_names: Dict[int, str],
                             log_dir: str = "performance_logs") -> PerformanceMonitor:
    """Factory function to create a performance monitor"""
    return PerformanceMonitor(
        action_class_names=action_class_names,
        severity_class_names=severity_class_names,
        log_dir=log_dir
    )


if __name__ == "__main__":
    # Example usage
    action_classes = {
        0: "Tackling", 1: "Standing tackling", 2: "High leg", 3: "Holding",
        4: "Pushing", 5: "Elbowing", 6: "Challenge", 7: "Dive"
    }
    
    severity_classes = {
        0: "No Offence", 1: "Offence + No Card", 
        2: "Offence + Yellow Card", 3: "Offence + Red Card"
    }
    
    monitor = create_performance_monitor(action_classes, severity_classes)
    
    # Simulate some metrics
    dummy_action_preds = torch.randn(32, 8)
    dummy_action_targets = torch.randint(0, 8, (32,))
    dummy_action_targets_onehot = torch.zeros(32, 8)
    dummy_action_targets_onehot.scatter_(1, dummy_action_targets.unsqueeze(1), 1)
    
    dummy_severity_preds = torch.randn(32, 4)
    dummy_severity_targets = torch.randint(0, 4, (32,))
    dummy_severity_targets_onehot = torch.zeros(32, 4)
    dummy_severity_targets_onehot.scatter_(1, dummy_severity_targets.unsqueeze(1), 1)
    
    # Update metrics
    epoch_metrics = monitor.update_metrics(
        epoch=1,
        action_predictions=dummy_action_preds,
        action_targets=dummy_action_targets_onehot,
        severity_predictions=dummy_severity_preds,
        severity_targets=dummy_severity_targets_onehot,
        loss_action=0.5,
        loss_severity=0.3,
        learning_rate=0.001
    )
    
    # Generate report
    report = monitor.generate_performance_report()
    print("Performance report generated successfully!")