"""
Adaptive Loss System with Performance-Based Optimization
Implements dynamic loss functions that adapt based on per-class performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassPerformanceMetrics:
    """Performance metrics for a specific class"""
    class_id: int
    recall: float
    precision: float
    f1_score: float
    support: int
    epochs_since_improvement: int = 0
    best_recall: float = 0.0
    recall_history: List[float] = None
    
    def __post_init__(self):
        if self.recall_history is None:
            self.recall_history = []

@dataclass
class LossConfig:
    """Configuration for adaptive loss parameters"""
    initial_gamma: float = 2.0
    initial_alpha: float = 1.0
    min_gamma: float = 0.5
    max_gamma: float = 5.0
    min_alpha: float = 0.1
    max_alpha: float = 10.0
    recall_threshold: float = 0.1  # 10% recall threshold
    weight_increase_factor: float = 1.5  # 50% increase
    adaptation_rate: float = 0.1
    history_window: int = 5


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with dynamic gamma and alpha parameters based on per-class recall.
    Implements requirement 4.1: dynamic gamma/alpha based on per-class recall.
    """
    
    def __init__(self, 
                 num_classes: int,
                 config: LossConfig = None,
                 class_names: List[str] = None,
                 label_smoothing: float = 0.0):
        """
        Initialize Adaptive Focal Loss.
        
        Args:
            num_classes: Number of classes
            config: Loss configuration parameters
            class_names: Optional class names for logging
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.config = config or LossConfig()
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.label_smoothing = label_smoothing
        
        # Initialize dynamic parameters
        self.register_buffer('gamma', torch.full((num_classes,), self.config.initial_gamma))
        self.register_buffer('alpha', torch.full((num_classes,), self.config.initial_alpha))
        self.register_buffer('class_weights', torch.ones(num_classes))
        
        # Performance tracking
        self.class_metrics = {}
        self.update_count = 0
        
        logger.info(f"AdaptiveFocalLoss initialized for {num_classes} classes")
        logger.info(f"Initial gamma: {self.config.initial_gamma}, alpha: {self.config.initial_alpha}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive focal loss.
        
        Args:
            inputs: Logits tensor of shape (batch_size, num_classes)
            targets: Target class indices of shape (batch_size,)
            
        Returns:
            Focal loss tensor
        """
        # Ensure all tensors are on the same device as inputs
        device = inputs.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth_targets = self._apply_label_smoothing(targets, inputs.size(1))
            log_probs = F.log_softmax(inputs, dim=1)
            loss = F.kl_div(log_probs, smooth_targets, reduction='none').sum(dim=1)
            pt = F.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard cross-entropy loss
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = F.nll_loss(log_probs, targets, reduction='none', weight=self.class_weights.to(inputs.device) if self.class_weights is not None else None)
            pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1))).squeeze(1)
            loss = ce_loss
        
        # Apply class-specific focal terms
        focal_weights = torch.zeros_like(loss)
        for i in range(self.num_classes):
            class_mask = (targets == i)
            if class_mask.any():
                gamma_i = self.gamma[i].to(inputs.device)
                alpha_i = self.alpha[i].to(inputs.device)
                focal_term = alpha_i * (1 - pt[class_mask]) ** gamma_i
                focal_weights[class_mask] = focal_term
        
        focal_loss = focal_weights * loss
        
        return focal_loss.mean()
    
    def _apply_label_smoothing(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Apply label smoothing to targets"""
        smooth_targets = torch.zeros(targets.size(0), num_classes, device=targets.device)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return smooth_targets
    
    def update_parameters(self, class_metrics: Dict[int, ClassPerformanceMetrics]):
        """
        Update gamma and alpha parameters based on class performance.
        
        Args:
            class_metrics: Dictionary mapping class_id to performance metrics
        """
        self.class_metrics = class_metrics
        self.update_count += 1
        
        for class_id, metrics in class_metrics.items():
            if class_id >= self.num_classes:
                continue
                
            recall = metrics.recall
            
            # Update gamma based on recall (higher gamma for lower recall)
            if recall < self.config.recall_threshold:
                # Increase gamma for hard classes
                target_gamma = self.config.max_gamma
                adaptation_strength = 1.0 - (recall / self.config.recall_threshold)
            else:
                # Decrease gamma for well-performing classes
                target_gamma = self.config.initial_gamma
                adaptation_strength = min(1.0, recall)
            
            # Smooth adaptation
            current_gamma = self.gamma[class_id].item()
            new_gamma = current_gamma + self.config.adaptation_rate * adaptation_strength * (target_gamma - current_gamma)
            new_gamma = torch.clamp(torch.tensor(new_gamma), self.config.min_gamma, self.config.max_gamma)
            self.gamma[class_id] = new_gamma
            
            # Update alpha based on recall and class difficulty
            if recall < self.config.recall_threshold:
                # Increase alpha for underperforming classes
                alpha_multiplier = self.config.weight_increase_factor * (1.0 + (self.config.recall_threshold - recall))
                new_alpha = min(self.config.max_alpha, self.config.initial_alpha * alpha_multiplier)
            else:
                # Gradually reduce alpha for well-performing classes
                new_alpha = max(self.config.min_alpha, self.config.initial_alpha * (1.0 - 0.1 * (recall - self.config.recall_threshold)))
            
            self.alpha[class_id] = new_alpha
        
        if self.update_count % 10 == 0:  # Log every 10 updates
            self._log_parameter_updates()
    
    def _log_parameter_updates(self):
        """Log current parameter values"""
        logger.info("Adaptive Focal Loss parameter update:")
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            recall = self.class_metrics.get(i, ClassPerformanceMetrics(i, 0.0, 0.0, 0.0, 0)).recall
            logger.info(f"  {class_name}: gamma={self.gamma[i]:.3f}, alpha={self.alpha[i]:.3f}, recall={recall:.3f}")
    
    def get_current_parameters(self) -> Dict:
        """Get current parameter values"""
        return {
            'gamma': self.gamma.cpu().numpy().tolist(),
            'alpha': self.alpha.cpu().numpy().tolist(),
            'class_weights': self.class_weights.cpu().numpy().tolist()
        }


class ClassBalanceLoss(nn.Module):
    """
    Class balance loss that dynamically adjusts weights based on class performance.
    Implements requirement 4.2: class balance loss component.
    """
    
    def __init__(self, 
                 num_classes: int,
                 initial_weights: Optional[torch.Tensor] = None,
                 adaptation_rate: float = 0.1):
        """
        Initialize Class Balance Loss.
        
        Args:
            num_classes: Number of classes
            initial_weights: Initial class weights
            adaptation_rate: Rate of weight adaptation
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate
        
        if initial_weights is not None:
            self.register_buffer('class_weights', initial_weights.clone())
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))
        
        # Track class frequencies for dynamic rebalancing
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        logger.info(f"ClassBalanceLoss initialized for {num_classes} classes")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class balance loss.
        
        Args:
            inputs: Logits tensor of shape (batch_size, num_classes)
            targets: Target class indices of shape (batch_size,)
            
        Returns:
            Class balance loss tensor
        """
        # Ensure all tensors are on the same device as inputs
        device = inputs.device
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
        if self.class_counts.device != device:
            self.class_counts = self.class_counts.to(device)
        if self.total_samples.device != device:
            self.total_samples = self.total_samples.to(device)
        # Update class counts
        self._update_class_counts(targets)
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        
        return loss
    
    def _update_class_counts(self, targets: torch.Tensor):
        """Update running class counts"""
        for class_id in range(self.num_classes):
            count = (targets == class_id).sum().float()
            self.class_counts[class_id] += count
            self.total_samples += count
    
    def update_weights(self, class_metrics: Dict[int, ClassPerformanceMetrics]):
        """
        Update class weights based on performance metrics.
        
        Args:
            class_metrics: Dictionary mapping class_id to performance metrics
        """
        new_weights = torch.zeros_like(self.class_weights)
        
        for class_id in range(self.num_classes):
            if class_id in class_metrics:
                metrics = class_metrics[class_id]
                recall = metrics.recall
                
                # Calculate inverse frequency weight
                if self.class_counts[class_id] > 0:
                    frequency = self.class_counts[class_id] / self.total_samples
                    inv_freq_weight = 1.0 / (frequency + 1e-8)
                else:
                    inv_freq_weight = 1.0
                
                # Combine with performance-based adjustment
                if recall < 0.1:  # 10% threshold
                    performance_multiplier = 2.0  # Double weight for poor performers
                elif recall < 0.3:  # 30% threshold
                    performance_multiplier = 1.5  # 50% increase
                else:
                    performance_multiplier = 1.0
                
                new_weight = inv_freq_weight * performance_multiplier
                new_weights[class_id] = new_weight
            else:
                new_weights[class_id] = self.class_weights[class_id]
        
        # Smooth weight updates
        self.class_weights = (1 - self.adaptation_rate) * self.class_weights + self.adaptation_rate * new_weights
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.mean()
    
    def get_current_weights(self) -> torch.Tensor:
        """Get current class weights"""
        return self.class_weights.clone()


class AttentionDiversityLoss(nn.Module):
    """
    Attention diversity loss to prevent attention collapse.
    Implements requirement for attention diversity loss component.
    """
    
    def __init__(self, diversity_weight: float = 0.1, similarity_threshold: float = 0.8):
        """
        Initialize Attention Diversity Loss.
        
        Args:
            diversity_weight: Weight for diversity loss component
            similarity_threshold: Threshold for considering attention patterns too similar
        """
        super().__init__()
        
        self.diversity_weight = diversity_weight
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"AttentionDiversityLoss initialized with weight={diversity_weight}")
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention diversity loss.
        
        Args:
            attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
                             or (batch_size, num_heads, seq_len)
            
        Returns:
            Diversity loss tensor
        """
        if attention_weights.dim() < 3:
            return torch.tensor(0.0, device=attention_weights.device)
        
        batch_size = attention_weights.shape[0]
        num_heads = attention_weights.shape[1]
        
        if num_heads < 2:
            return torch.tensor(0.0, device=attention_weights.device)
        
        # Flatten attention patterns for each head
        flattened_attention = attention_weights.view(batch_size, num_heads, -1)
        
        # Normalize attention patterns
        normalized_attention = F.normalize(flattened_attention, p=2, dim=-1)
        
        # Compute pairwise cosine similarities between heads
        similarities = torch.matmul(normalized_attention, normalized_attention.transpose(-2, -1))
        
        # Remove diagonal (self-similarity) and take upper triangular part
        mask = torch.triu(torch.ones(num_heads, num_heads, device=similarities.device), diagonal=1).bool()
        similarities_upper = similarities[:, mask]
        
        # Penalize high similarities (encourage diversity)
        diversity_loss = torch.relu(similarities_upper - self.similarity_threshold).mean()
        
        return self.diversity_weight * diversity_loss


class GradientBalancer:
    """
    Gradient balancing mechanism to ensure fair gradient flow across classes.
    Implements requirement 4.4: gradient balancing mechanisms across classes.
    """
    
    def __init__(self, 
                 num_classes: int,
                 balance_method: str = 'magnitude',
                 adaptation_rate: float = 0.1):
        """
        Initialize Gradient Balancer.
        
        Args:
            num_classes: Number of classes
            balance_method: Method for balancing ('magnitude', 'norm', 'adaptive')
            adaptation_rate: Rate of adaptation for balancing factors
        """
        self.num_classes = num_classes
        self.balance_method = balance_method
        self.adaptation_rate = adaptation_rate
        
        # Track gradient statistics
        self.gradient_magnitudes = defaultdict(lambda: deque(maxlen=100))
        self.balance_factors = torch.ones(num_classes)
        
        logger.info(f"GradientBalancer initialized with method={balance_method}")
    
    def balance_gradients(self, 
                         model: nn.Module,
                         loss_per_class: torch.Tensor,
                         class_metrics: Dict[int, ClassPerformanceMetrics]) -> torch.Tensor:
        """
        Balance gradients across classes.
        
        Args:
            model: Model to balance gradients for
            loss_per_class: Loss tensor per class
            class_metrics: Performance metrics per class
            
        Returns:
            Balanced loss tensor
        """
        if self.balance_method == 'magnitude':
            return self._balance_by_magnitude(model, loss_per_class)
        elif self.balance_method == 'norm':
            return self._balance_by_norm(model, loss_per_class)
        elif self.balance_method == 'adaptive':
            return self._balance_adaptive(model, loss_per_class, class_metrics)
        else:
            return loss_per_class.mean()
    
    def _balance_by_magnitude(self, model: nn.Module, loss_per_class: torch.Tensor) -> torch.Tensor:
        """Balance gradients by magnitude"""
        # Compute gradients for each class loss
        class_gradients = []
        
        for i, class_loss in enumerate(loss_per_class):
            if class_loss.requires_grad:
                # Compute gradients for this class
                grad_outputs = torch.autograd.grad(
                    class_loss, 
                    model.parameters(), 
                    retain_graph=True, 
                    create_graph=True,
                    allow_unused=True
                )
                
                # Calculate gradient magnitude
                grad_magnitude = 0.0
                for grad in grad_outputs:
                    if grad is not None:
                        grad_magnitude += grad.norm().item()
                
                self.gradient_magnitudes[i].append(grad_magnitude)
                class_gradients.append(grad_magnitude)
            else:
                class_gradients.append(0.0)
        
        # Calculate balancing weights
        if len(class_gradients) > 0 and max(class_gradients) > 0:
            max_grad = max(class_gradients)
            balance_weights = torch.tensor([max_grad / (grad + 1e-8) for grad in class_gradients])
            balance_weights = torch.clamp(balance_weights, 0.1, 10.0)  # Limit extreme weights
        else:
            balance_weights = torch.ones(len(loss_per_class))
        
        # Apply balancing weights
        balanced_loss = (loss_per_class * balance_weights).mean()
        
        return balanced_loss
    
    def _balance_by_norm(self, model: nn.Module, loss_per_class: torch.Tensor) -> torch.Tensor:
        """Balance gradients by L2 norm"""
        class_grad_norms = []
        
        for i, class_loss in enumerate(loss_per_class):
            if class_loss.requires_grad:
                # Compute gradients for this class
                grad_outputs = torch.autograd.grad(
                    class_loss, 
                    model.parameters(), 
                    retain_graph=True, 
                    create_graph=True,
                    allow_unused=True
                )
                
                # Calculate L2 norm of gradients
                grad_norm = 0.0
                for grad in grad_outputs:
                    if grad is not None:
                        grad_norm += grad.pow(2).sum().item()
                grad_norm = math.sqrt(grad_norm)
                
                class_grad_norms.append(grad_norm)
            else:
                class_grad_norms.append(0.0)
        
        # Calculate balancing weights based on inverse norm
        if len(class_grad_norms) > 0 and max(class_grad_norms) > 0:
            mean_norm = np.mean([norm for norm in class_grad_norms if norm > 0])
            balance_weights = torch.tensor([mean_norm / (norm + 1e-8) for norm in class_grad_norms])
            balance_weights = torch.clamp(balance_weights, 0.1, 10.0)
        else:
            balance_weights = torch.ones(len(loss_per_class))
        
        balanced_loss = (loss_per_class * balance_weights).mean()
        
        return balanced_loss
    
    def _balance_adaptive(self, 
                         model: nn.Module, 
                         loss_per_class: torch.Tensor,
                         class_metrics: Dict[int, ClassPerformanceMetrics]) -> torch.Tensor:
        """Adaptive gradient balancing based on class performance"""
        balance_weights = torch.ones(len(loss_per_class))
        
        for i in range(len(loss_per_class)):
            if i in class_metrics:
                recall = class_metrics[i].recall
                
                # Increase weight for underperforming classes
                if recall < 0.1:
                    balance_weights[i] = 3.0  # Triple weight for very poor classes
                elif recall < 0.3:
                    balance_weights[i] = 2.0  # Double weight for poor classes
                elif recall > 0.8:
                    balance_weights[i] = 0.5  # Reduce weight for well-performing classes
        
        balanced_loss = (loss_per_class * balance_weights).mean()
        
        return balanced_loss
    
    def get_gradient_statistics(self) -> Dict:
        """Get current gradient statistics"""
        stats = {}
        for class_id, magnitudes in self.gradient_magnitudes.items():
            if len(magnitudes) > 0:
                stats[class_id] = {
                    'mean_magnitude': np.mean(magnitudes),
                    'std_magnitude': np.std(magnitudes),
                    'recent_magnitude': magnitudes[-1] if magnitudes else 0.0
                }
        return stats


class DynamicLossSystem(nn.Module):
    """
    Dynamic loss system that combines multiple loss components and adapts based on performance.
    Implements requirements 4.1, 4.2, 4.3, 4.4: comprehensive adaptive loss system.
    """
    
    def __init__(self,
                 num_action_classes: int,
                 num_severity_classes: int,
                 config: LossConfig = None,
                 action_class_names: List[str] = None,
                 severity_class_names: List[str] = None):
        """
        Initialize Dynamic Loss System.
        
        Args:
            num_action_classes: Number of action classes
            num_severity_classes: Number of severity classes
            config: Loss configuration
            action_class_names: Names of action classes
            severity_class_names: Names of severity classes
        """
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_severity_classes = num_severity_classes
        self.config = config or LossConfig()
        
        # Initialize loss components
        self.action_focal_loss = AdaptiveFocalLoss(
            num_action_classes, 
            config=self.config,
            class_names=action_class_names
        )
        
        self.severity_focal_loss = AdaptiveFocalLoss(
            num_severity_classes,
            config=self.config, 
            class_names=severity_class_names
        )
        
        self.action_balance_loss = ClassBalanceLoss(num_action_classes)
        self.severity_balance_loss = ClassBalanceLoss(num_severity_classes)
        
        self.attention_diversity_loss = AttentionDiversityLoss()
        
        # Gradient balancers
        self.action_gradient_balancer = GradientBalancer(num_action_classes)
        self.severity_gradient_balancer = GradientBalancer(num_severity_classes)
        
        # Loss combination weights
        self.register_buffer('focal_weight', torch.tensor(1.0))
        self.register_buffer('balance_weight', torch.tensor(0.5))
        self.register_buffer('diversity_weight', torch.tensor(0.1))
        
        logger.info(f"DynamicLossSystem initialized:")
        logger.info(f"  Action classes: {num_action_classes}")
        logger.info(f"  Severity classes: {num_severity_classes}")
    
    def forward(self, 
                action_logits: torch.Tensor,
                severity_logits: torch.Tensor,
                action_targets: torch.Tensor,
                severity_targets: torch.Tensor,
                attention_info: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined adaptive loss.
        
        Args:
            action_logits: Action prediction logits
            severity_logits: Severity prediction logits
            action_targets: Action target labels
            severity_targets: Severity target labels
            attention_info: Optional attention information for diversity loss
            
        Returns:
            Combined loss tensor and loss component dictionary
        """
        loss_components = {}
        
        # Compute focal losses
        action_focal = self.action_focal_loss(action_logits, action_targets)
        severity_focal = self.severity_focal_loss(severity_logits, severity_targets)
        loss_components['action_focal'] = action_focal
        loss_components['severity_focal'] = severity_focal
        
        # Compute balance losses
        action_balance = self.action_balance_loss(action_logits, action_targets)
        severity_balance = self.severity_balance_loss(severity_logits, severity_targets)
        loss_components['action_balance'] = action_balance
        loss_components['severity_balance'] = severity_balance
        
        # Compute attention diversity loss
        diversity_loss = torch.tensor(0.0, device=action_logits.device)
        if attention_info is not None and 'diversity_loss' in attention_info:
            diversity_loss = attention_info['diversity_loss']
        loss_components['diversity'] = diversity_loss
        
        # Combine losses
        total_loss = (
            self.focal_weight * (action_focal + severity_focal) +
            self.balance_weight * (action_balance + severity_balance) +
            self.diversity_weight * diversity_loss
        )
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components
    
    def update_from_metrics(self, 
                           action_metrics: Dict[int, ClassPerformanceMetrics],
                           severity_metrics: Dict[int, ClassPerformanceMetrics]):
        """
        Update loss parameters based on performance metrics.
        
        Args:
            action_metrics: Action class performance metrics
            severity_metrics: Severity class performance metrics
        """
        # Update focal loss parameters
        self.action_focal_loss.update_parameters(action_metrics)
        self.severity_focal_loss.update_parameters(severity_metrics)
        
        # Update balance loss weights
        self.action_balance_loss.update_weights(action_metrics)
        self.severity_balance_loss.update_weights(severity_metrics)
        
        # Adapt loss combination weights based on overall performance
        self._adapt_combination_weights(action_metrics, severity_metrics)
    
    def _adapt_combination_weights(self,
                                  action_metrics: Dict[int, ClassPerformanceMetrics],
                                  severity_metrics: Dict[int, ClassPerformanceMetrics]):
        """Adapt the weights for combining different loss components"""
        # Calculate average recalls
        action_recalls = [m.recall for m in action_metrics.values()]
        severity_recalls = [m.recall for m in severity_metrics.values()]
        
        avg_action_recall = np.mean(action_recalls) if action_recalls else 0.0
        avg_severity_recall = np.mean(severity_recalls) if severity_recalls else 0.0
        
        # Increase focal weight if overall performance is poor
        if avg_action_recall < 0.3 or avg_severity_recall < 0.3:
            self.focal_weight = torch.tensor(1.5)  # Increase focal loss importance
            self.balance_weight = torch.tensor(0.8)  # Increase balance loss importance
        elif avg_action_recall > 0.7 and avg_severity_recall > 0.7:
            self.focal_weight = torch.tensor(0.8)  # Reduce focal loss importance
            self.balance_weight = torch.tensor(0.3)  # Reduce balance loss importance
        
        # Increase diversity weight if attention collapse is detected
        # This would be determined by analyzing attention patterns
        self.diversity_weight = torch.tensor(0.1)  # Keep constant for now
    
    def get_current_config(self) -> Dict:
        """Get current loss configuration"""
        return {
            'focal_weight': self.focal_weight.item(),
            'balance_weight': self.balance_weight.item(),
            'diversity_weight': self.diversity_weight.item(),
            'action_focal_params': self.action_focal_loss.get_current_parameters(),
            'severity_focal_params': self.severity_focal_loss.get_current_parameters(),
            'action_balance_weights': self.action_balance_loss.get_current_weights().cpu().numpy().tolist(),
            'severity_balance_weights': self.severity_balance_loss.get_current_weights().cpu().numpy().tolist()
        }


if __name__ == "__main__":
    print("Testing Adaptive Loss System...")
    
    # Test configuration
    num_action_classes = 8
    num_severity_classes = 4
    batch_size = 4
    
    # Initialize system
    loss_system = DynamicLossSystem(
        num_action_classes=num_action_classes,
        num_severity_classes=num_severity_classes
    )
    print("✓ DynamicLossSystem initialized")
    
    # Create dummy data
    action_logits = torch.randn(batch_size, num_action_classes, requires_grad=True)
    severity_logits = torch.randn(batch_size, num_severity_classes, requires_grad=True)
    action_targets = torch.randint(0, num_action_classes, (batch_size,))
    severity_targets = torch.randint(0, num_severity_classes, (batch_size,))
    
    # Test forward pass
    try:
        total_loss, loss_components = loss_system(
            action_logits, severity_logits, action_targets, severity_targets
        )
        print(f"✓ Forward pass successful")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Loss components: {list(loss_components.keys())}")
        
        # Test backward pass
        total_loss.backward()
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Forward/backward pass failed: {e}")
        raise
    
    # Test parameter updates
    try:
        # Create dummy metrics
        action_metrics = {
            i: ClassPerformanceMetrics(i, recall=0.05 if i in [4, 7] else 0.6, precision=0.5, f1_score=0.3, support=10)
            for i in range(num_action_classes)
        }
        severity_metrics = {
            i: ClassPerformanceMetrics(i, recall=0.08 if i == 3 else 0.5, precision=0.4, f1_score=0.3, support=10)
            for i in range(num_severity_classes)
        }
        
        loss_system.update_from_metrics(action_metrics, severity_metrics)
        print("✓ Parameter update successful")
        
        # Check parameter changes
        config = loss_system.get_current_config()
        print(f"  Updated focal weight: {config['focal_weight']:.3f}")
        print(f"  Updated balance weight: {config['balance_weight']:.3f}")
        
    except Exception as e:
        print(f"✗ Parameter update failed: {e}")
        raise
    
    print("\n✓ All tests passed!")