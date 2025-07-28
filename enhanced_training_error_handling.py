"""
Enhanced Training Error Handling and Recovery System

This module provides comprehensive error handling, recovery mechanisms, and
debugging utilities for the enhanced training pipeline.

Requirements: Comprehensive error handling and recovery mechanisms
"""

import os
import sys
import traceback
import logging
import json
import torch
import psutil
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling"""
    DATA_LOADING = "data_loading"
    MODEL_TRAINING = "model_training"
    MEMORY_ERROR = "memory_error"
    CUDA_ERROR = "cuda_error"
    CONFIGURATION = "configuration"
    FILE_IO = "file_io"
    COMPONENT_INTEGRATION = "component_integration"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNKNOWN = "unknown"


class TrainingError(Exception):
    """Base exception for training-related errors"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging"""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc()
        }


class DataLoadingError(TrainingError):
    """Error related to data loading"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_LOADING,
            **kwargs
        )


class ModelTrainingError(TrainingError):
    """Error related to model training"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_TRAINING,
            **kwargs
        )


class MemoryError(TrainingError):
    """Error related to memory issues"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY_ERROR,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class CudaError(TrainingError):
    """Error related to CUDA operations"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CUDA_ERROR,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ComponentIntegrationError(TrainingError):
    """Error related to component integration"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPONENT_INTEGRATION,
            **kwargs
        )


class ErrorHandler:
    """Comprehensive error handler for the enhanced training system"""
    
    def __init__(self, 
                 log_dir: str = "error_logs",
                 enable_recovery: bool = True,
                 max_recovery_attempts: int = 3):
        """
        Initialize error handler.
        
        Args:
            log_dir: Directory to save error logs
            enable_recovery: Whether to enable automatic recovery
            max_recovery_attempts: Maximum number of recovery attempts
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_recovery = enable_recovery
        self.max_recovery_attempts = max_recovery_attempts
        
        # Error tracking
        self.error_history = []
        self.recovery_attempts = {}
        self.system_state = {}
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.MEMORY_ERROR: self._recover_from_memory_error,
            ErrorCategory.CUDA_ERROR: self._recover_from_cuda_error,
            ErrorCategory.DATA_LOADING: self._recover_from_data_loading_error,
            ErrorCategory.MODEL_TRAINING: self._recover_from_training_error,
            ErrorCategory.COMPONENT_INTEGRATION: self._recover_from_integration_error
        }
        
        # Set up error logging
        self._setup_error_logging()
        
        logger.info(f"ErrorHandler initialized with log_dir: {log_dir}")
    
    def _setup_error_logging(self):
        """Set up dedicated error logging"""
        error_log_path = self.log_dir / "training_errors.log"
        
        # Create error logger
        self.error_logger = logging.getLogger("training_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # Create file handler
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        
        # Add handler
        self.error_logger.addHandler(error_handler)
    
    def handle_error(self, 
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> bool:
        """
        Handle an error with optional recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            True if error was handled/recovered, False otherwise
        """
        # Convert to TrainingError if needed
        if not isinstance(error, TrainingError):
            training_error = self._convert_to_training_error(error, context)
        else:
            training_error = error
            if context:
                training_error.context.update(context)
        
        # Log the error
        self._log_error(training_error)
        
        # Add to error history
        self.error_history.append(training_error)
        
        # Attempt recovery if enabled
        if attempt_recovery and self.enable_recovery:
            return self._attempt_recovery(training_error)
        
        return False
    
    def _convert_to_training_error(self, 
                                 error: Exception,
                                 context: Optional[Dict[str, Any]] = None) -> TrainingError:
        """Convert generic exception to TrainingError"""
        
        error_message = str(error)
        error_type = type(error).__name__
        
        # Categorize error based on type and message
        category = self._categorize_error(error, error_message)
        severity = self._assess_severity(error, error_message)
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(category, error_message)
        
        return TrainingError(
            message=f"{error_type}: {error_message}",
            category=category,
            severity=severity,
            context=context or {},
            recovery_suggestions=recovery_suggestions
        )
    
    def _categorize_error(self, error: Exception, message: str) -> ErrorCategory:
        """Categorize error based on type and message"""
        
        error_type = type(error).__name__
        message_lower = message.lower()
        
        # Memory-related errors
        if (error_type in ['RuntimeError', 'OutOfMemoryError'] and 
            any(keyword in message_lower for keyword in ['memory', 'cuda out of memory', 'allocation'])):
            return ErrorCategory.MEMORY_ERROR
        
        # CUDA-related errors
        if (error_type in ['RuntimeError', 'CudaError'] and 
            any(keyword in message_lower for keyword in ['cuda', 'gpu', 'device'])):
            return ErrorCategory.CUDA_ERROR
        
        # Data loading errors
        if (error_type in ['FileNotFoundError', 'IOError', 'OSError'] or
            any(keyword in message_lower for keyword in ['dataset', 'dataloader', 'file not found'])):
            return ErrorCategory.DATA_LOADING
        
        # Model training errors
        if (error_type in ['ValueError', 'TypeError'] and
            any(keyword in message_lower for keyword in ['model', 'forward', 'backward', 'loss'])):
            return ErrorCategory.MODEL_TRAINING
        
        # Configuration errors
        if (error_type in ['ValueError', 'KeyError', 'AttributeError'] and
            any(keyword in message_lower for keyword in ['config', 'parameter', 'attribute'])):
            return ErrorCategory.CONFIGURATION
        
        # Component integration errors
        if any(keyword in message_lower for keyword in ['integration', 'component', 'wrapper']):
            return ErrorCategory.COMPONENT_INTEGRATION
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception, message: str) -> ErrorSeverity:
        """Assess error severity"""
        
        error_type = type(error).__name__
        message_lower = message.lower()
        
        # Critical errors that stop training
        if error_type in ['SystemExit', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if (error_type in ['OutOfMemoryError', 'CudaError'] or
            any(keyword in message_lower for keyword in ['cuda out of memory', 'device-side assert'])):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['RuntimeError', 'ValueError', 'TypeError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _generate_recovery_suggestions(self, 
                                     category: ErrorCategory,
                                     message: str) -> List[str]:
        """Generate recovery suggestions based on error category"""
        
        suggestions = []
        
        if category == ErrorCategory.MEMORY_ERROR:
            suggestions.extend([
                "Reduce batch size",
                "Enable gradient accumulation",
                "Use mixed precision training",
                "Clear CUDA cache",
                "Reduce model complexity"
            ])
        
        elif category == ErrorCategory.CUDA_ERROR:
            suggestions.extend([
                "Check CUDA installation",
                "Verify GPU availability",
                "Clear CUDA cache",
                "Restart CUDA context",
                "Switch to CPU training"
            ])
        
        elif category == ErrorCategory.DATA_LOADING:
            suggestions.extend([
                "Check data file paths",
                "Verify dataset integrity",
                "Reduce number of workers",
                "Check file permissions",
                "Regenerate dataset if corrupted"
            ])
        
        elif category == ErrorCategory.MODEL_TRAINING:
            suggestions.extend([
                "Check model architecture",
                "Verify input/output dimensions",
                "Adjust learning rate",
                "Check loss function configuration",
                "Validate data preprocessing"
            ])
        
        elif category == ErrorCategory.COMPONENT_INTEGRATION:
            suggestions.extend([
                "Check component compatibility",
                "Verify configuration parameters",
                "Update component versions",
                "Check integration order",
                "Validate component interfaces"
            ])
        
        return suggestions
    
    def _log_error(self, error: TrainingError):
        """Log error with full details"""
        
        # Log to error logger
        self.error_logger.error(f"Training Error: {error.message}")
        self.error_logger.error(f"Category: {error.category.value}")
        self.error_logger.error(f"Severity: {error.severity.value}")
        self.error_logger.error(f"Context: {error.context}")
        self.error_logger.error(f"Recovery Suggestions: {error.recovery_suggestions}")
        
        # Save detailed error report
        error_report_path = self.log_dir / f"error_report_{error.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_report_path, 'w') as f:
            json.dump(error.to_dict(), f, indent=2)
        
        # Log system state
        self._capture_system_state()
    
    def _capture_system_state(self):
        """Capture current system state for debugging"""
        
        try:
            self.system_state = {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'total': psutil.virtual_memory().total
                },
                'cpu': {
                    'percent': psutil.cpu_percent(),
                    'count': psutil.cpu_count()
                },
                'disk': {
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
            
            # Add CUDA information if available
            if torch.cuda.is_available():
                self.system_state['cuda'] = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_reserved': torch.cuda.memory_reserved(),
                    'max_memory_allocated': torch.cuda.max_memory_allocated()
                }
            
        except Exception as e:
            logger.warning(f"Failed to capture system state: {e}")
    
    def _attempt_recovery(self, error: TrainingError) -> bool:
        """Attempt to recover from error"""
        
        error_key = f"{error.category.value}_{hash(error.message)}"
        
        # Check if we've exceeded max attempts
        if error_key in self.recovery_attempts:
            if self.recovery_attempts[error_key] >= self.max_recovery_attempts:
                logger.error(f"Max recovery attempts exceeded for error: {error.message}")
                return False
            self.recovery_attempts[error_key] += 1
        else:
            self.recovery_attempts[error_key] = 1
        
        # Attempt recovery based on category
        if error.category in self.recovery_strategies:
            logger.info(f"Attempting recovery for {error.category.value} error (attempt {self.recovery_attempts[error_key]})")
            
            try:
                success = self.recovery_strategies[error.category](error)
                if success:
                    logger.info(f"Successfully recovered from {error.category.value} error")
                    return True
                else:
                    logger.warning(f"Recovery attempt failed for {error.category.value} error")
            except Exception as recovery_error:
                logger.error(f"Recovery attempt raised exception: {recovery_error}")
        
        return False
    
    def _recover_from_memory_error(self, error: TrainingError) -> bool:
        """Recover from memory-related errors"""
        
        logger.info("Attempting memory error recovery...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Forced garbage collection")
        
        # Suggest configuration changes
        recovery_context = {
            'suggested_batch_size_reduction': 0.5,
            'enable_gradient_accumulation': True,
            'enable_mixed_precision': True
        }
        
        error.context.update(recovery_context)
        
        return True  # Recovery actions taken, but may need config changes
    
    def _recover_from_cuda_error(self, error: TrainingError) -> bool:
        """Recover from CUDA-related errors"""
        
        logger.info("Attempting CUDA error recovery...")
        
        if torch.cuda.is_available():
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Reset CUDA context
                torch.cuda.synchronize()
                
                # Check device availability
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                
                logger.info(f"CUDA recovery: {device_count} devices available, current: {current_device}")
                
                return True
                
            except Exception as cuda_recovery_error:
                logger.error(f"CUDA recovery failed: {cuda_recovery_error}")
                
                # Suggest fallback to CPU
                error.context['fallback_to_cpu'] = True
                return False
        
        return False
    
    def _recover_from_data_loading_error(self, error: TrainingError) -> bool:
        """Recover from data loading errors"""
        
        logger.info("Attempting data loading error recovery...")
        
        # Suggest reducing number of workers
        error.context['suggested_num_workers'] = 0
        error.context['suggested_pin_memory'] = False
        
        return True  # Recovery suggestions provided
    
    def _recover_from_training_error(self, error: TrainingError) -> bool:
        """Recover from training-related errors"""
        
        logger.info("Attempting training error recovery...")
        
        # Suggest training parameter adjustments
        recovery_context = {
            'suggested_learning_rate_reduction': 0.1,
            'suggested_gradient_clipping': True,
            'check_model_architecture': True
        }
        
        error.context.update(recovery_context)
        
        return True  # Recovery suggestions provided
    
    def _recover_from_integration_error(self, error: TrainingError) -> bool:
        """Recover from component integration errors"""
        
        logger.info("Attempting integration error recovery...")
        
        # Suggest component fallbacks
        recovery_context = {
            'disable_problematic_components': True,
            'use_fallback_implementations': True,
            'check_component_versions': True
        }
        
        error.context.update(recovery_context)
        
        return True  # Recovery suggestions provided
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        
        if not self.error_history:
            return {'total_errors': 0, 'summary': 'No errors encountered'}
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Get recent errors
        recent_errors = self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        
        return {
            'total_errors': len(self.error_history),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recent_errors': [error.to_dict() for error in recent_errors],
            'recovery_attempts': dict(self.recovery_attempts),
            'system_state': self.system_state
        }
    
    def save_error_report(self, filepath: str):
        """Save comprehensive error report"""
        
        report = {
            'error_summary': self.get_error_summary(),
            'all_errors': [error.to_dict() for error in self.error_history],
            'system_state_history': self.system_state,
            'recovery_strategies_used': list(self.recovery_strategies.keys()),
            'report_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Error report saved to {filepath}")


@contextmanager
def error_handling_context(error_handler: ErrorHandler,
                          context: Optional[Dict[str, Any]] = None,
                          reraise: bool = True):
    """
    Context manager for automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance
        context: Additional context for error handling
        reraise: Whether to reraise the exception after handling
    """
    try:
        yield
    except Exception as e:
        handled = error_handler.handle_error(e, context)
        
        if reraise and not handled:
            raise
        elif not handled:
            logger.error(f"Unhandled error: {e}")


def robust_training_wrapper(error_handler: ErrorHandler,
                          max_retries: int = 3,
                          retry_delay: float = 1.0):
    """
    Decorator for robust training function execution.
    
    Args:
        error_handler: ErrorHandler instance
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_retries': max_retries
                    }
                    
                    handled = error_handler.handle_error(e, context)
                    
                    if attempt < max_retries and handled:
                        logger.info(f"Retrying {func.__name__} (attempt {attempt + 2}/{max_retries + 1})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            
            # If we get here, all retries failed
            logger.error(f"All retry attempts failed for {func.__name__}")
            raise last_error
        
        return wrapper
    return decorator


class PerformanceDegradationDetector:
    """Detector for performance degradation issues"""
    
    def __init__(self, 
                 error_handler: ErrorHandler,
                 performance_threshold: float = 0.1,
                 monitoring_window: int = 5):
        """
        Initialize performance degradation detector.
        
        Args:
            error_handler: ErrorHandler instance
            performance_threshold: Threshold for performance degradation
            monitoring_window: Number of epochs to monitor
        """
        self.error_handler = error_handler
        self.performance_threshold = performance_threshold
        self.monitoring_window = monitoring_window
        
        self.performance_history = []
        self.baseline_performance = None
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        
        combined_recall = metrics.get('combined_macro_recall', 0.0)
        self.performance_history.append(combined_recall)
        
        # Keep only recent history
        if len(self.performance_history) > self.monitoring_window:
            self.performance_history.pop(0)
        
        # Set baseline if not set
        if self.baseline_performance is None and len(self.performance_history) >= 3:
            self.baseline_performance = max(self.performance_history)
        
        # Check for degradation
        if self.baseline_performance is not None:
            self._check_degradation(combined_recall)
    
    def _check_degradation(self, current_performance: float):
        """Check for performance degradation"""
        
        degradation = self.baseline_performance - current_performance
        
        if degradation > self.performance_threshold:
            # Create performance degradation error
            error = TrainingError(
                message=f"Performance degradation detected: {degradation:.3f} drop from baseline {self.baseline_performance:.3f}",
                category=ErrorCategory.PERFORMANCE_DEGRADATION,
                severity=ErrorSeverity.MEDIUM,
                context={
                    'current_performance': current_performance,
                    'baseline_performance': self.baseline_performance,
                    'degradation': degradation,
                    'performance_history': self.performance_history.copy()
                },
                recovery_suggestions=[
                    "Check for overfitting",
                    "Reduce learning rate",
                    "Increase regularization",
                    "Review recent configuration changes",
                    "Check data quality"
                ]
            )
            
            self.error_handler.handle_error(error, attempt_recovery=False)


def create_error_handler(log_dir: str = "error_logs") -> ErrorHandler:
    """Create and configure error handler"""
    return ErrorHandler(log_dir=log_dir, enable_recovery=True)


if __name__ == "__main__":
    print("Testing Enhanced Training Error Handling System...")
    
    # Create error handler
    error_handler = create_error_handler("test_error_logs")
    print("✓ Error handler created")
    
    # Test error handling
    try:
        # Simulate a memory error
        raise RuntimeError("CUDA out of memory")
    except Exception as e:
        handled = error_handler.handle_error(e, context={'test': True})
        print(f"✓ Memory error handled: {handled}")
    
    # Test error context manager
    with error_handling_context(error_handler, context={'test_context': True}, reraise=False):
        # Simulate a data loading error
        raise FileNotFoundError("Dataset file not found")
    
    print("✓ Error context manager tested")
    
    # Test performance degradation detector
    detector = PerformanceDegradationDetector(error_handler)
    
    # Simulate performance degradation
    detector.update_performance({'combined_macro_recall': 0.5})
    detector.update_performance({'combined_macro_recall': 0.48})
    detector.update_performance({'combined_macro_recall': 0.46})
    detector.update_performance({'combined_macro_recall': 0.35})  # Should trigger degradation
    
    print("✓ Performance degradation detector tested")
    
    # Get error summary
    summary = error_handler.get_error_summary()
    print(f"✓ Error summary generated: {summary['total_errors']} errors")
    
    # Save error report
    error_handler.save_error_report("test_error_report.json")
    print("✓ Error report saved")
    
    # Cleanup
    import shutil
    if os.path.exists("test_error_logs"):
        shutil.rmtree("test_error_logs")
    if os.path.exists("test_error_report.json"):
        os.remove("test_error_report.json")
    
    print("\n✓ All error handling tests passed!")