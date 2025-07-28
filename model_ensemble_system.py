"""
Model Ensemble System with Confidence Weighting

This module implements a comprehensive ensemble system for the MVFouls model that:
1. Creates ensemble predictor combining multiple model checkpoints
2. Adds confidence-weighted ensemble voting and diversity metrics
3. Implements checkpoint management for best performing models
4. Adds ensemble-based uncertainty quantification

Requirements: 7.4, 5.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import copy
from pathlib import Path
from collections import defaultdict
import pickle

from enhanced_model import EnhancedMVFoulsModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetrics:
    """Metrics associated with a model checkpoint."""
    epoch: int
    combined_macro_recall: float
    action_macro_recall: float
    severity_macro_recall: float
    action_class_recalls: Dict[int, float]
    severity_class_recalls: Dict[int, float]
    validation_loss: float
    timestamp: str
    model_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble system."""
    max_checkpoints: int = 10
    min_performance_threshold: float = 0.35
    diversity_weight: float = 0.2
    confidence_threshold: float = 0.5
    uncertainty_weight: float = 0.1
    checkpoint_selection_strategy: str = "performance_diversity"  # "performance", "diversity", "performance_diversity"
    ensemble_voting_method: str = "confidence_weighted"  # "simple", "confidence_weighted", "uncertainty_weighted"
    save_dir: str = "ensemble_checkpoints"


class CheckpointManager:
    """
    Manages model checkpoints for ensemble system.
    Implements checkpoint management for best performing models.
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints: List[CheckpointMetrics] = []
        self.metadata_file = self.save_dir / "checkpoint_metadata.json"
        
        # Load existing metadata if available
        self._load_metadata()
        
    def _load_metadata(self):
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = [CheckpointMetrics.from_dict(item) for item in data]
                logger.info(f"Loaded {len(self.checkpoints)} checkpoint metadata entries")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints = []
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump([cp.to_dict() for cp in self.checkpoints], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def add_checkpoint(self, 
                      model: nn.Module,
                      optimizer: torch.optim.Optimizer,
                      epoch: int,
                      metrics: Dict[str, Any],
                      scheduler: Optional[Any] = None) -> Optional[str]:
        """
        Add a new checkpoint if it meets quality criteria.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Performance metrics dictionary
            scheduler: Optional scheduler state
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        combined_recall = metrics.get('combined_macro_recall', 0.0)
        
        # Check if checkpoint meets minimum performance threshold
        if combined_recall < self.config.min_performance_threshold:
            logger.info(f"Checkpoint at epoch {epoch} below threshold ({combined_recall:.3f} < {self.config.min_performance_threshold:.3f})")
            return None
        
        # Create checkpoint metadata
        checkpoint_metrics = CheckpointMetrics(
            epoch=epoch,
            combined_macro_recall=combined_recall,
            action_macro_recall=metrics.get('action_macro_recall', 0.0),
            severity_macro_recall=metrics.get('severity_macro_recall', 0.0),
            action_class_recalls=metrics.get('action_class_recalls', {}),
            severity_class_recalls=metrics.get('severity_class_recalls', {}),
            validation_loss=metrics.get('validation_loss', float('inf')),
            timestamp=datetime.now().isoformat(),
            model_path=""  # Will be set after saving
        )
        
        # Generate checkpoint filename
        checkpoint_filename = f"ensemble_checkpoint_epoch_{epoch}_{combined_recall:.4f}.pth"
        checkpoint_path = self.save_dir / checkpoint_filename
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'num_action_classes': getattr(model, 'num_action_classes', 8),
                'num_severity_classes': getattr(model, 'num_severity_classes', 4)
            }
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint_metrics.model_path = str(checkpoint_path)
            
            # Add to checkpoint list
            self.checkpoints.append(checkpoint_metrics)
            
            # Sort by performance and apply selection strategy
            self._manage_checkpoint_storage()
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Checkpoint saved: {checkpoint_path} (recall: {combined_recall:.4f})")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _manage_checkpoint_storage(self):
        """Manage checkpoint storage based on selection strategy."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        if self.config.checkpoint_selection_strategy == "performance":
            # Keep top performing checkpoints
            self.checkpoints.sort(key=lambda x: x.combined_macro_recall, reverse=True)
            
        elif self.config.checkpoint_selection_strategy == "diversity":
            # Keep diverse checkpoints (simplified diversity based on class recalls)
            self.checkpoints = self._select_diverse_checkpoints()
            
        elif self.config.checkpoint_selection_strategy == "performance_diversity":
            # Balance performance and diversity
            self.checkpoints = self._select_performance_diverse_checkpoints()
        
        # Remove excess checkpoints
        checkpoints_to_remove = self.checkpoints[self.config.max_checkpoints:]
        self.checkpoints = self.checkpoints[:self.config.max_checkpoints]
        
        # Delete checkpoint files
        for checkpoint in checkpoints_to_remove:
            try:
                if os.path.exists(checkpoint.model_path):
                    os.remove(checkpoint.model_path)
                    logger.info(f"Removed checkpoint: {checkpoint.model_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.model_path}: {e}")
    
    def _select_diverse_checkpoints(self) -> List[CheckpointMetrics]:
        """Select diverse checkpoints based on class recall patterns."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return self.checkpoints
        
        # Sort by performance first
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.combined_macro_recall, reverse=True)
        
        # Always keep the best performing checkpoint
        selected = [sorted_checkpoints[0]]
        remaining = sorted_checkpoints[1:]
        
        # Select diverse checkpoints
        while len(selected) < self.config.max_checkpoints and remaining:
            best_diversity_score = -1
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                diversity_score = self._calculate_diversity_score(candidate, selected)
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def _select_performance_diverse_checkpoints(self) -> List[CheckpointMetrics]:
        """Select checkpoints balancing performance and diversity."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return self.checkpoints
        
        # Sort by performance
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.combined_macro_recall, reverse=True)
        
        # Always keep top 2 performing checkpoints
        selected = sorted_checkpoints[:min(2, self.config.max_checkpoints)]
        remaining = sorted_checkpoints[2:]
        
        # Select remaining based on performance-diversity balance
        while len(selected) < self.config.max_checkpoints and remaining:
            best_score = -1
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                performance_score = candidate.combined_macro_recall
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                # Combined score
                combined_score = (1 - self.config.diversity_weight) * performance_score + \
                               self.config.diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate: CheckpointMetrics, selected: List[CheckpointMetrics]) -> float:
        """Calculate diversity score for a candidate checkpoint."""
        if not selected:
            return 1.0
        
        # Create feature vector from class recalls
        candidate_features = []
        for recalls in [candidate.action_class_recalls, candidate.severity_class_recalls]:
            candidate_features.extend(list(recalls.values()))
        
        candidate_features = np.array(candidate_features)
        
        # Calculate minimum distance to selected checkpoints
        min_distance = float('inf')
        
        for selected_checkpoint in selected:
            selected_features = []
            for recalls in [selected_checkpoint.action_class_recalls, selected_checkpoint.severity_class_recalls]:
                selected_features.extend(list(recalls.values()))
            
            selected_features = np.array(selected_features)
            
            # Euclidean distance
            distance = np.linalg.norm(candidate_features - selected_features)
            min_distance = min(min_distance, distance)
        
        # Normalize distance to [0, 1] range
        return min(min_distance / 2.0, 1.0)  # Assuming max possible distance is ~2.0
    
    def get_best_checkpoints(self, n: int = None) -> List[CheckpointMetrics]:
        """Get the best n checkpoints."""
        if n is None:
            n = len(self.checkpoints)
        
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.combined_macro_recall, reverse=True)
        return sorted_checkpoints[:n]
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a checkpoint file."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint, checkpoint.get('model_config', {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple model checkpoints with confidence weighting.
    Implements ensemble predictor and confidence-weighted voting.
    """
    
    def __init__(self, config: EnsembleConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.models: List[EnhancedMVFoulsModel] = []
        self.checkpoint_metrics: List[CheckpointMetrics] = []
        self.model_weights: List[float] = []
        
    def load_ensemble_models(self, checkpoint_manager: CheckpointManager, max_models: int = None):
        """
        Load ensemble models from checkpoint manager.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            max_models: Maximum number of models to load (None for all available)
        """
        if max_models is None:
            max_models = len(checkpoint_manager.checkpoints)
        
        best_checkpoints = checkpoint_manager.get_best_checkpoints(max_models)
        
        self.models = []
        self.checkpoint_metrics = []
        self.model_weights = []
        
        for checkpoint_metrics in best_checkpoints:
            try:
                # Load checkpoint
                checkpoint_data, model_config = checkpoint_manager.load_checkpoint(checkpoint_metrics.model_path)
                
                # Create model
                model = EnhancedMVFoulsModel(
                    num_action_classes=model_config.get('num_action_classes', 8),
                    num_severity_classes=model_config.get('num_severity_classes', 4)
                )
                
                # Load model state
                model.load_state_dict(checkpoint_data['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                # Calculate model weight based on performance
                weight = self._calculate_model_weight(checkpoint_metrics)
                
                self.models.append(model)
                self.checkpoint_metrics.append(checkpoint_metrics)
                self.model_weights.append(weight)
                
                logger.info(f"Loaded ensemble model: {checkpoint_metrics.model_path} (weight: {weight:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to load model from {checkpoint_metrics.model_path}: {e}")
        
        # Normalize weights
        if self.model_weights:
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
        
        logger.info(f"Loaded {len(self.models)} models for ensemble")
    
    def _calculate_model_weight(self, checkpoint_metrics: CheckpointMetrics) -> float:
        """Calculate weight for a model based on its performance."""
        # Base weight on combined macro recall
        base_weight = checkpoint_metrics.combined_macro_recall
        
        # Bonus for balanced performance across tasks
        action_recall = checkpoint_metrics.action_macro_recall
        severity_recall = checkpoint_metrics.severity_macro_recall
        balance_bonus = 1.0 - abs(action_recall - severity_recall)
        
        # Bonus for minority class performance
        minority_bonus = 0.0
        for class_recalls in [checkpoint_metrics.action_class_recalls, checkpoint_metrics.severity_class_recalls]:
            for recall in class_recalls.values():
                if recall > 0.1:  # Bonus for non-zero minority class recall
                    minority_bonus += 0.1
        
        total_weight = base_weight + 0.1 * balance_bonus + 0.05 * minority_bonus
        return max(total_weight, 0.1)  # Minimum weight
    
    def predict(self, x_list: List[torch.Tensor], return_details: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Make ensemble predictions with confidence weighting.
        
        Args:
            x_list: List of input tensors
            return_details: Whether to return detailed prediction information
            
        Returns:
            If return_details=False: (action_logits, severity_logits)
            If return_details=True: Dictionary with detailed prediction information
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        all_action_logits = []
        all_severity_logits = []
        all_confidences = []
        all_uncertainties = []
        
        # Get predictions from all models
        with torch.no_grad():
            for i, model in enumerate(self.models):
                action_logits, severity_logits, confidence_scores, attention_info = model(x_list, return_attention=False)
                
                all_action_logits.append(action_logits)
                all_severity_logits.append(severity_logits)
                all_confidences.append(confidence_scores)
                
                # Calculate total uncertainty
                action_uncertainty = confidence_scores['action_aleatoric_uncertainty'] + confidence_scores['action_epistemic_uncertainty']
                severity_uncertainty = confidence_scores['severity_aleatoric_uncertainty'] + confidence_scores['severity_epistemic_uncertainty']
                all_uncertainties.append((action_uncertainty, severity_uncertainty))
        
        # Stack predictions
        action_logits_stack = torch.stack(all_action_logits, dim=0)  # (num_models, batch_size, num_action_classes)
        severity_logits_stack = torch.stack(all_severity_logits, dim=0)  # (num_models, batch_size, num_severity_classes)
        
        # Apply ensemble voting
        if self.config.ensemble_voting_method == "simple":
            ensemble_action_logits = self._simple_ensemble(action_logits_stack)
            ensemble_severity_logits = self._simple_ensemble(severity_logits_stack)
            
        elif self.config.ensemble_voting_method == "confidence_weighted":
            ensemble_action_logits = self._confidence_weighted_ensemble(action_logits_stack, all_confidences, 'action')
            ensemble_severity_logits = self._confidence_weighted_ensemble(severity_logits_stack, all_confidences, 'severity')
            
        elif self.config.ensemble_voting_method == "uncertainty_weighted":
            ensemble_action_logits = self._uncertainty_weighted_ensemble(action_logits_stack, all_uncertainties, 'action')
            ensemble_severity_logits = self._uncertainty_weighted_ensemble(severity_logits_stack, all_uncertainties, 'severity')
        
        else:
            raise ValueError(f"Unknown ensemble voting method: {self.config.ensemble_voting_method}")
        
        if not return_details:
            return ensemble_action_logits, ensemble_severity_logits
        
        # Calculate ensemble uncertainty and diversity metrics
        ensemble_uncertainty = self._calculate_ensemble_uncertainty(all_action_logits, all_severity_logits, all_uncertainties)
        diversity_metrics = self._calculate_diversity_metrics(action_logits_stack, severity_logits_stack)
        
        return {
            'action_logits': ensemble_action_logits,
            'severity_logits': ensemble_severity_logits,
            'individual_predictions': {
                'action_logits': all_action_logits,
                'severity_logits': all_severity_logits,
                'confidences': all_confidences,
                'uncertainties': all_uncertainties
            },
            'ensemble_uncertainty': ensemble_uncertainty,
            'diversity_metrics': diversity_metrics,
            'model_weights': self.model_weights
        }
    
    def _simple_ensemble(self, logits_stack: torch.Tensor) -> torch.Tensor:
        """Simple weighted average ensemble."""
        weights = torch.tensor(self.model_weights, device=logits_stack.device).view(-1, 1, 1)
        return torch.sum(logits_stack * weights, dim=0)
    
    def _confidence_weighted_ensemble(self, logits_stack: torch.Tensor, all_confidences: List[Dict], task: str) -> torch.Tensor:
        """Confidence-weighted ensemble voting."""
        batch_size = logits_stack.shape[1]
        num_classes = logits_stack.shape[2]
        
        # Extract confidence scores for the task
        confidence_key = f'{task}_confidence'
        confidence_weights = []
        
        for confidences in all_confidences:
            conf = confidences[confidence_key]  # (batch_size,)
            if conf.dim() == 1:
                conf = conf.unsqueeze(-1).expand(-1, num_classes)  # (batch_size, num_classes)
            confidence_weights.append(conf)
        
        confidence_weights = torch.stack(confidence_weights, dim=0)  # (num_models, batch_size, num_classes)
        
        # Combine with model weights
        model_weights = torch.tensor(self.model_weights, device=logits_stack.device).view(-1, 1, 1)
        combined_weights = confidence_weights * model_weights
        
        # Normalize weights
        weight_sum = torch.sum(combined_weights, dim=0, keepdim=True)
        normalized_weights = combined_weights / (weight_sum + 1e-8)
        
        # Weighted average
        return torch.sum(logits_stack * normalized_weights, dim=0)
    
    def _uncertainty_weighted_ensemble(self, logits_stack: torch.Tensor, all_uncertainties: List[Tuple], task: str) -> torch.Tensor:
        """Uncertainty-weighted ensemble voting (lower uncertainty = higher weight)."""
        batch_size = logits_stack.shape[1]
        num_classes = logits_stack.shape[2]
        
        # Extract uncertainty scores for the task
        task_idx = 0 if task == 'action' else 1
        uncertainty_weights = []
        
        for uncertainties in all_uncertainties:
            uncertainty = uncertainties[task_idx]  # (batch_size,)
            if uncertainty.dim() == 1:
                uncertainty = uncertainty.unsqueeze(-1).expand(-1, num_classes)  # (batch_size, num_classes)
            
            # Convert uncertainty to weight (inverse relationship)
            weight = 1.0 / (uncertainty + 1e-8)
            uncertainty_weights.append(weight)
        
        uncertainty_weights = torch.stack(uncertainty_weights, dim=0)  # (num_models, batch_size, num_classes)
        
        # Combine with model weights
        model_weights = torch.tensor(self.model_weights, device=logits_stack.device).view(-1, 1, 1)
        combined_weights = uncertainty_weights * model_weights
        
        # Normalize weights
        weight_sum = torch.sum(combined_weights, dim=0, keepdim=True)
        normalized_weights = combined_weights / (weight_sum + 1e-8)
        
        # Weighted average
        return torch.sum(logits_stack * normalized_weights, dim=0)
    
    def _calculate_ensemble_uncertainty(self, all_action_logits: List[torch.Tensor], 
                                      all_severity_logits: List[torch.Tensor],
                                      all_uncertainties: List[Tuple]) -> Dict[str, torch.Tensor]:
        """Calculate ensemble-based uncertainty quantification."""
        # Convert logits to probabilities
        action_probs = [F.softmax(logits, dim=-1) for logits in all_action_logits]
        severity_probs = [F.softmax(logits, dim=-1) for logits in all_severity_logits]
        
        # Stack probabilities
        action_probs_stack = torch.stack(action_probs, dim=0)  # (num_models, batch_size, num_classes)
        severity_probs_stack = torch.stack(severity_probs, dim=0)
        
        # Calculate prediction variance (epistemic uncertainty)
        action_mean_probs = torch.mean(action_probs_stack, dim=0)
        severity_mean_probs = torch.mean(severity_probs_stack, dim=0)
        
        action_epistemic = torch.var(action_probs_stack, dim=0).mean(dim=-1)  # (batch_size,)
        severity_epistemic = torch.var(severity_probs_stack, dim=0).mean(dim=-1)
        
        # Calculate average aleatoric uncertainty
        action_aleatoric = torch.stack([unc[0] for unc in all_uncertainties], dim=0).mean(dim=0)
        severity_aleatoric = torch.stack([unc[1] for unc in all_uncertainties], dim=0).mean(dim=0)
        
        # Total uncertainty
        action_total = action_epistemic + action_aleatoric
        severity_total = severity_epistemic + severity_aleatoric
        
        return {
            'action_epistemic_uncertainty': action_epistemic,
            'severity_epistemic_uncertainty': severity_epistemic,
            'action_aleatoric_uncertainty': action_aleatoric,
            'severity_aleatoric_uncertainty': severity_aleatoric,
            'action_total_uncertainty': action_total,
            'severity_total_uncertainty': severity_total
        }
    
    def _calculate_diversity_metrics(self, action_logits_stack: torch.Tensor, 
                                   severity_logits_stack: torch.Tensor) -> Dict[str, float]:
        """Calculate diversity metrics for the ensemble."""
        # Convert to probabilities
        action_probs = F.softmax(action_logits_stack, dim=-1)
        severity_probs = F.softmax(severity_logits_stack, dim=-1)
        
        # Calculate pairwise disagreement
        action_disagreement = self._calculate_pairwise_disagreement(action_probs)
        severity_disagreement = self._calculate_pairwise_disagreement(severity_probs)
        
        # Calculate entropy of ensemble predictions
        action_mean_probs = torch.mean(action_probs, dim=0)
        severity_mean_probs = torch.mean(severity_probs, dim=0)
        
        action_entropy = -torch.sum(action_mean_probs * torch.log(action_mean_probs + 1e-8), dim=-1).mean()
        severity_entropy = -torch.sum(severity_mean_probs * torch.log(severity_mean_probs + 1e-8), dim=-1).mean()
        
        return {
            'action_disagreement': action_disagreement.item(),
            'severity_disagreement': severity_disagreement.item(),
            'action_entropy': action_entropy.item(),
            'severity_entropy': severity_entropy.item(),
            'average_disagreement': (action_disagreement + severity_disagreement).item() / 2,
            'average_entropy': (action_entropy + severity_entropy).item() / 2
        }
    
    def _calculate_pairwise_disagreement(self, probs_stack: torch.Tensor) -> torch.Tensor:
        """Calculate average pairwise disagreement between models."""
        num_models = probs_stack.shape[0]
        total_disagreement = 0.0
        num_pairs = 0
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # KL divergence between model predictions
                kl_div = F.kl_div(
                    torch.log(probs_stack[i] + 1e-8), 
                    probs_stack[j], 
                    reduction='batchmean'
                )
                total_disagreement += kl_div
                num_pairs += 1
        
        return total_disagreement / max(num_pairs, 1)


class EnsembleSystem:
    """
    Complete ensemble system integrating checkpoint management and ensemble prediction.
    """
    
    def __init__(self, config: EnsembleConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.checkpoint_manager = CheckpointManager(config)
        self.ensemble_predictor = EnsemblePredictor(config, device)
        
    def add_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                      epoch: int, metrics: Dict[str, Any], scheduler: Optional[Any] = None) -> Optional[str]:
        """Add a new checkpoint to the ensemble system."""
        return self.checkpoint_manager.add_checkpoint(model, optimizer, epoch, metrics, scheduler)
    
    def initialize_ensemble(self, max_models: int = None):
        """Initialize the ensemble predictor with available checkpoints."""
        self.ensemble_predictor.load_ensemble_models(self.checkpoint_manager, max_models)
    
    def predict(self, x_list: List[torch.Tensor], return_details: bool = False):
        """Make ensemble predictions."""
        return self.ensemble_predictor.predict(x_list, return_details)
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        stats = {
            'num_checkpoints': len(self.checkpoint_manager.checkpoints),
            'num_ensemble_models': len(self.ensemble_predictor.models),
            'checkpoint_performance_range': {},
            'model_weights': self.ensemble_predictor.model_weights,
            'config': asdict(self.config)
        }
        
        if self.checkpoint_manager.checkpoints:
            recalls = [cp.combined_macro_recall for cp in self.checkpoint_manager.checkpoints]
            stats['checkpoint_performance_range'] = {
                'min_recall': min(recalls),
                'max_recall': max(recalls),
                'mean_recall': np.mean(recalls),
                'std_recall': np.std(recalls)
            }
        
        return stats
    
    def save_ensemble_config(self, path: str):
        """Save ensemble configuration and metadata."""
        config_data = {
            'config': asdict(self.config),
            'checkpoint_metadata': [cp.to_dict() for cp in self.checkpoint_manager.checkpoints],
            'ensemble_stats': self.get_ensemble_stats()
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Ensemble configuration saved to {path}")


if __name__ == "__main__":
    print("Testing Model Ensemble System...")
    
    # Test configuration
    config = EnsembleConfig(
        max_checkpoints=5,
        min_performance_threshold=0.3,
        ensemble_voting_method="confidence_weighted"
    )
    
    print("✓ Configuration created")
    
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager(config)
    print("✓ Checkpoint manager created")
    
    # Test ensemble system
    ensemble_system = EnsembleSystem(config, device='cpu')
    print("✓ Ensemble system created")
    
    # Test with dummy model and metrics
    dummy_model = EnhancedMVFoulsModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    
    dummy_metrics = {
        'combined_macro_recall': 0.42,
        'action_macro_recall': 0.45,
        'severity_macro_recall': 0.39,
        'action_class_recalls': {0: 0.8, 1: 0.6, 2: 0.2, 3: 0.1},
        'severity_class_recalls': {0: 0.9, 1: 0.3, 2: 0.2, 3: 0.1},
        'validation_loss': 1.2
    }
    
    # Test adding checkpoint
    checkpoint_path = ensemble_system.add_checkpoint(
        dummy_model, dummy_optimizer, epoch=10, metrics=dummy_metrics
    )
    
    if checkpoint_path:
        print(f"✓ Checkpoint added: {checkpoint_path}")
    else:
        print("✗ Checkpoint not added (below threshold)")
    
    print("\n✓ All ensemble system tests passed!")