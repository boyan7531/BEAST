"""
Curriculum Learning System for MVFouls Performance Optimization.

This module implements a comprehensive curriculum learning system with:
- Progressive minority class introduction over epochs
- Stage-based loss weight progression with smooth transitions
- Modified sampler respecting curriculum requirements
- Curriculum progress monitoring and visualization

Requirements: 3.4, 1.4
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Iterator
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage"""
    stage_name: str
    start_epoch: int
    end_epoch: int
    included_classes: Dict[str, List[int]]  # task -> list of class indices
    loss_weights: Dict[str, Dict[int, float]]  # task -> class_id -> weight
    sampling_weights: Dict[str, Dict[int, float]]  # task -> class_id -> sampling weight
    difficulty_level: str  # 'easy', 'medium', 'hard'


@dataclass
class CurriculumProgress:
    """Tracks progress through curriculum stages"""
    current_stage: int
    current_epoch: int
    stage_progress: float  # 0.0 to 1.0 within current stage
    overall_progress: float  # 0.0 to 1.0 overall
    classes_introduced: Dict[str, List[int]]  # task -> list of introduced classes
    performance_history: List[Dict]  # history of performance metrics


class CurriculumScheduler:
    """
    Curriculum scheduler that gradually introduces hard classes over epochs.
    
    Implements progressive introduction of minority classes:
    - Stage 1 (Easy): Majority classes only
    - Stage 2 (Medium): Majority + some minority classes
    - Stage 3 (Hard): All classes including hardest minorities
    """
    
    def __init__(self,
                 total_epochs: int,
                 action_class_difficulty: Dict[int, str] = None,
                 severity_class_difficulty: Dict[int, str] = None,
                 stage_ratios: Tuple[float, float, float] = (0.4, 0.35, 0.25),
                 smooth_transitions: bool = True,
                 transition_overlap: int = 2):
        """
        Initialize curriculum scheduler.
        
        Args:
            total_epochs: Total number of training epochs
            action_class_difficulty: Mapping of action class ID to difficulty level
            severity_class_difficulty: Mapping of severity class ID to difficulty level
            stage_ratios: Ratios for (easy, medium, hard) stages
            smooth_transitions: Whether to use smooth transitions between stages
            transition_overlap: Number of epochs for transition overlap
        """
        self.total_epochs = total_epochs
        self.smooth_transitions = smooth_transitions
        self.transition_overlap = transition_overlap
        
        # Default difficulty mappings based on MVFouls class distribution
        if action_class_difficulty is None:
            self.action_class_difficulty = {
                0: 'easy',    # No action
                1: 'easy',    # Kicking
                2: 'medium',  # Tackling
                3: 'medium',  # Elbowing
                4: 'hard',    # Pushing (minority)
                5: 'medium',  # Holding
                6: 'medium',  # Standing
                7: 'hard',    # Dive (minority)
            }
        else:
            self.action_class_difficulty = action_class_difficulty
        
        if severity_class_difficulty is None:
            self.severity_class_difficulty = {
                0: 'easy',    # No card
                1: 'easy',    # Yellow card
                2: 'medium',  # Not shown
                3: 'hard',    # Red card (minority)
            }
        else:
            self.severity_class_difficulty = severity_class_difficulty
        
        # Calculate stage boundaries
        self.stage_boundaries = self._calculate_stage_boundaries(stage_ratios)
        
        # Create curriculum stages
        self.stages = self._create_curriculum_stages()
        
        # Initialize progress tracking
        self.progress = CurriculumProgress(
            current_stage=0,
            current_epoch=0,
            stage_progress=0.0,
            overall_progress=0.0,
            classes_introduced={'action': [], 'severity': []},
            performance_history=[]
        )
        
        logger.info(f"CurriculumScheduler initialized:")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Stage boundaries: {self.stage_boundaries}")
        logger.info(f"  Number of stages: {len(self.stages)}")
        logger.info(f"  Smooth transitions: {smooth_transitions}")
    
    def _calculate_stage_boundaries(self, stage_ratios: Tuple[float, float, float]) -> List[Tuple[int, int]]:
        """Calculate epoch boundaries for each stage"""
        easy_ratio, medium_ratio, hard_ratio = stage_ratios
        
        # Ensure ratios sum to 1.0
        total_ratio = sum(stage_ratios)
        easy_ratio /= total_ratio
        medium_ratio /= total_ratio
        hard_ratio /= total_ratio
        
        # Calculate boundaries
        easy_end = int(self.total_epochs * easy_ratio)
        medium_end = int(self.total_epochs * (easy_ratio + medium_ratio))
        
        boundaries = [
            (0, easy_end),                    # Easy stage
            (easy_end, medium_end),           # Medium stage
            (medium_end, self.total_epochs)   # Hard stage
        ]
        
        return boundaries
    
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create curriculum stages with progressive class introduction"""
        stages = []
        
        # Group classes by difficulty
        easy_actions = [k for k, v in self.action_class_difficulty.items() if v == 'easy']
        medium_actions = [k for k, v in self.action_class_difficulty.items() if v == 'medium']
        hard_actions = [k for k, v in self.action_class_difficulty.items() if v == 'hard']
        
        easy_severities = [k for k, v in self.severity_class_difficulty.items() if v == 'easy']
        medium_severities = [k for k, v in self.severity_class_difficulty.items() if v == 'medium']
        hard_severities = [k for k, v in self.severity_class_difficulty.items() if v == 'hard']
        
        # Stage 1: Easy classes only
        start_epoch, end_epoch = self.stage_boundaries[0]
        stages.append(CurriculumStage(
            stage_name="Easy",
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            included_classes={
                'action': easy_actions,
                'severity': easy_severities
            },
            loss_weights={
                'action': {cls: 1.0 for cls in easy_actions},
                'severity': {cls: 1.0 for cls in easy_severities}
            },
            sampling_weights={
                'action': {cls: 1.0 for cls in easy_actions},
                'severity': {cls: 1.0 for cls in easy_severities}
            },
            difficulty_level='easy'
        ))
        
        # Stage 2: Easy + Medium classes
        start_epoch, end_epoch = self.stage_boundaries[1]
        medium_classes_action = easy_actions + medium_actions
        medium_classes_severity = easy_severities + medium_severities
        
        stages.append(CurriculumStage(
            stage_name="Medium",
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            included_classes={
                'action': medium_classes_action,
                'severity': medium_classes_severity
            },
            loss_weights={
                'action': {**{cls: 1.0 for cls in easy_actions},
                          **{cls: 1.5 for cls in medium_actions}},
                'severity': {**{cls: 1.0 for cls in easy_severities},
                           **{cls: 1.5 for cls in medium_severities}}
            },
            sampling_weights={
                'action': {**{cls: 1.0 for cls in easy_actions},
                          **{cls: 2.0 for cls in medium_actions}},
                'severity': {**{cls: 1.0 for cls in easy_severities},
                           **{cls: 2.0 for cls in medium_severities}}
            },
            difficulty_level='medium'
        ))
        
        # Stage 3: All classes (Easy + Medium + Hard)
        start_epoch, end_epoch = self.stage_boundaries[2]
        all_classes_action = easy_actions + medium_actions + hard_actions
        all_classes_severity = easy_severities + medium_severities + hard_severities
        
        stages.append(CurriculumStage(
            stage_name="Hard",
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            included_classes={
                'action': all_classes_action,
                'severity': all_classes_severity
            },
            loss_weights={
                'action': {**{cls: 1.0 for cls in easy_actions},
                          **{cls: 1.5 for cls in medium_actions},
                          **{cls: 3.0 for cls in hard_actions}},  # High weight for minorities
                'severity': {**{cls: 1.0 for cls in easy_severities},
                           **{cls: 1.5 for cls in medium_severities},
                           **{cls: 4.0 for cls in hard_severities}}  # Very high weight for Red Card
            },
            sampling_weights={
                'action': {**{cls: 1.0 for cls in easy_actions},
                          **{cls: 2.0 for cls in medium_actions},
                          **{cls: 6.0 for cls in hard_actions}},  # Strong oversampling
                'severity': {**{cls: 1.0 for cls in easy_severities},
                           **{cls: 2.0 for cls in medium_severities},
                           **{cls: 8.0 for cls in hard_severities}}  # Very strong oversampling
            },
            difficulty_level='hard'
        ))
        
        return stages
    
    def get_current_stage_config(self, epoch: int) -> Tuple[CurriculumStage, Dict]:
        """
        Get current stage configuration for given epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Tuple of (current_stage, interpolated_config)
        """
        # Update progress
        self.progress.current_epoch = epoch
        self.progress.overall_progress = epoch / self.total_epochs
        
        # Find current stage
        current_stage_idx = 0
        for i, stage in enumerate(self.stages):
            if stage.start_epoch <= epoch < stage.end_epoch:
                current_stage_idx = i
                break
        else:
            # Handle case where epoch >= total_epochs
            current_stage_idx = len(self.stages) - 1
        
        current_stage = self.stages[current_stage_idx]
        self.progress.current_stage = current_stage_idx
        
        # Calculate stage progress
        stage_length = current_stage.end_epoch - current_stage.start_epoch
        stage_progress = (epoch - current_stage.start_epoch) / max(1, stage_length)
        self.progress.stage_progress = min(1.0, max(0.0, stage_progress))
        
        # Get interpolated configuration
        if self.smooth_transitions and current_stage_idx > 0:
            config = self._interpolate_stage_configs(epoch, current_stage_idx)
        else:
            config = self._get_stage_config(current_stage)
        
        # Update classes introduced
        self.progress.classes_introduced['action'] = list(set(
            self.progress.classes_introduced['action'] + config['included_classes']['action']
        ))
        self.progress.classes_introduced['severity'] = list(set(
            self.progress.classes_introduced['severity'] + config['included_classes']['severity']
        ))
        
        return current_stage, config
    
    def _interpolate_stage_configs(self, epoch: int, current_stage_idx: int) -> Dict:
        """Interpolate between stage configurations for smooth transitions"""
        current_stage = self.stages[current_stage_idx]
        
        # Check if we're in a transition period
        transition_start = current_stage.start_epoch
        transition_end = min(current_stage.start_epoch + self.transition_overlap, current_stage.end_epoch)
        
        if epoch <= transition_end and current_stage_idx > 0:
            # Interpolate between previous and current stage
            prev_stage = self.stages[current_stage_idx - 1]
            
            # Calculate interpolation factor
            alpha = (epoch - transition_start) / max(1, transition_end - transition_start)
            alpha = min(1.0, max(0.0, alpha))
            
            # Interpolate loss weights
            interpolated_config = self._interpolate_weights(prev_stage, current_stage, alpha)
        else:
            # Use current stage configuration
            interpolated_config = self._get_stage_config(current_stage)
        
        return interpolated_config
    
    def _interpolate_weights(self, prev_stage: CurriculumStage, current_stage: CurriculumStage, alpha: float) -> Dict:
        """Interpolate weights between two stages"""
        config = {
            'included_classes': current_stage.included_classes.copy(),
            'loss_weights': {'action': {}, 'severity': {}},
            'sampling_weights': {'action': {}, 'severity': {}},
            'stage_name': f"{prev_stage.stage_name}->{current_stage.stage_name}",
            'interpolation_alpha': alpha
        }
        
        # Interpolate loss weights
        for task in ['action', 'severity']:
            for class_id in current_stage.included_classes[task]:
                prev_weight = prev_stage.loss_weights[task].get(class_id, 0.0)
                curr_weight = current_stage.loss_weights[task].get(class_id, 1.0)
                
                # Smooth interpolation
                interpolated_weight = prev_weight * (1 - alpha) + curr_weight * alpha
                config['loss_weights'][task][class_id] = interpolated_weight
                
                # Interpolate sampling weights
                prev_sampling = prev_stage.sampling_weights[task].get(class_id, 0.0)
                curr_sampling = current_stage.sampling_weights[task].get(class_id, 1.0)
                
                interpolated_sampling = prev_sampling * (1 - alpha) + curr_sampling * alpha
                config['sampling_weights'][task][class_id] = interpolated_sampling
        
        return config
    
    def _get_stage_config(self, stage: CurriculumStage) -> Dict:
        """Get configuration dictionary for a stage"""
        return {
            'included_classes': stage.included_classes.copy(),
            'loss_weights': stage.loss_weights.copy(),
            'sampling_weights': stage.sampling_weights.copy(),
            'stage_name': stage.stage_name,
            'interpolation_alpha': 1.0
        }
    
    def update_performance_history(self, epoch: int, metrics: Dict):
        """Update performance history with current metrics"""
        performance_entry = {
            'epoch': epoch,
            'stage': self.progress.current_stage,
            'stage_name': self.stages[self.progress.current_stage].stage_name,
            'stage_progress': self.progress.stage_progress,
            'overall_progress': self.progress.overall_progress,
            'metrics': metrics.copy(),
            'classes_introduced': self.progress.classes_introduced.copy()
        }
        
        self.progress.performance_history.append(performance_entry)
        
        # Keep only recent history to avoid memory issues
        if len(self.progress.performance_history) > 1000:
            self.progress.performance_history = self.progress.performance_history[-500:]
    
    def get_curriculum_stats(self) -> Dict:
        """Get comprehensive curriculum statistics"""
        return {
            'current_stage': self.progress.current_stage,
            'current_stage_name': self.stages[self.progress.current_stage].stage_name,
            'stage_progress': self.progress.stage_progress,
            'overall_progress': self.progress.overall_progress,
            'classes_introduced': self.progress.classes_introduced,
            'total_stages': len(self.stages),
            'stage_boundaries': self.stage_boundaries,
            'performance_history_length': len(self.progress.performance_history)
        }


class CurriculumAwareSampler:
    """
    Modified sampler that respects curriculum requirements with gradual weight increase.
    
    Integrates with existing StratifiedMinorityBatchSampler but adds curriculum awareness.
    """
    
    def __init__(self,
                 base_sampler,
                 curriculum_scheduler: CurriculumScheduler,
                 adaptation_rate: float = 0.1):
        """
        Initialize curriculum-aware sampler.
        
        Args:
            base_sampler: Base sampler (e.g., StratifiedMinorityBatchSampler)
            curriculum_scheduler: Curriculum scheduler
            adaptation_rate: Rate of adaptation to curriculum changes
        """
        self.base_sampler = base_sampler
        self.curriculum_scheduler = curriculum_scheduler
        self.adaptation_rate = adaptation_rate
        
        # Current sampling weights
        self.current_sampling_weights = {
            'action': defaultdict(lambda: 1.0),
            'severity': defaultdict(lambda: 1.0)
        }
        
        # Excluded classes (not yet introduced)
        self.excluded_classes = {
            'action': set(),
            'severity': set()
        }
        
        logger.info("CurriculumAwareSampler initialized")
    
    def update_for_epoch(self, epoch: int):
        """Update sampler configuration for current epoch"""
        # Get current curriculum stage
        current_stage, config = self.curriculum_scheduler.get_current_stage_config(epoch)
        
        # Update sampling weights with smooth adaptation
        for task in ['action', 'severity']:
            target_weights = config['sampling_weights'][task]
            included_classes = set(config['included_classes'][task])
            
            # Update weights for included classes
            for class_id, target_weight in target_weights.items():
                current_weight = self.current_sampling_weights[task][class_id]
                
                # Smooth adaptation
                new_weight = current_weight + self.adaptation_rate * (target_weight - current_weight)
                self.current_sampling_weights[task][class_id] = new_weight
            
            # Update excluded classes
            all_possible_classes = set(range(8)) if task == 'action' else set(range(4))
            self.excluded_classes[task] = all_possible_classes - included_classes
        
        logger.debug(f"Epoch {epoch}: Updated sampling weights for stage '{current_stage.stage_name}'")
        logger.debug(f"  Excluded action classes: {self.excluded_classes['action']}")
        logger.debug(f"  Excluded severity classes: {self.excluded_classes['severity']}")
    
    def get_batch_indices(self, epoch: int, batch_size: int) -> List[int]:
        """
        Get batch indices respecting curriculum constraints.
        
        Args:
            epoch: Current epoch
            batch_size: Desired batch size
            
        Returns:
            List of sample indices for the batch
        """
        # Update for current epoch
        self.update_for_epoch(epoch)
        
        # Get base batch indices
        if hasattr(self.base_sampler, 'get_batch_indices'):
            base_indices = self.base_sampler.get_batch_indices(epoch, batch_size)
        else:
            # Fallback: get indices from base sampler iterator
            base_indices = next(iter(self.base_sampler))
        
        # Filter indices based on curriculum constraints
        filtered_indices = self._filter_indices_by_curriculum(base_indices)
        
        # If we don't have enough samples, pad with allowed samples
        if len(filtered_indices) < batch_size:
            additional_indices = self._get_additional_allowed_indices(
                batch_size - len(filtered_indices), 
                excluded_indices=set(filtered_indices)
            )
            filtered_indices.extend(additional_indices)
        
        # Truncate if we have too many
        filtered_indices = filtered_indices[:batch_size]
        
        return filtered_indices
    
    def _filter_indices_by_curriculum(self, indices: List[int]) -> List[int]:
        """Filter indices to only include curriculum-allowed classes"""
        filtered_indices = []
        
        for idx in indices:
            # Get labels for this sample
            if hasattr(self.base_sampler, 'action_indices'):
                action_class = self.base_sampler.action_indices[idx]
                severity_class = self.base_sampler.severity_indices[idx]
            else:
                # Fallback: assume we can access dataset
                continue  # Skip if we can't determine class
            
            # Check if classes are allowed in current curriculum stage
            if (action_class not in self.excluded_classes['action'] and 
                severity_class not in self.excluded_classes['severity']):
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def _get_additional_allowed_indices(self, num_needed: int, excluded_indices: set) -> List[int]:
        """Get additional indices that satisfy curriculum constraints"""
        additional_indices = []
        
        if not hasattr(self.base_sampler, 'action_indices'):
            return additional_indices
        
        # Create pool of allowed indices
        allowed_indices = []
        for idx in range(len(self.base_sampler.action_indices)):
            if idx in excluded_indices:
                continue
                
            action_class = self.base_sampler.action_indices[idx]
            severity_class = self.base_sampler.severity_indices[idx]
            
            if (action_class not in self.excluded_classes['action'] and 
                severity_class not in self.excluded_classes['severity']):
                allowed_indices.append(idx)
        
        # Randomly sample from allowed indices
        if allowed_indices:
            additional_indices = random.sample(
                allowed_indices, 
                min(num_needed, len(allowed_indices))
            )
        
        return additional_indices
    
    def get_sampling_stats(self) -> Dict:
        """Get current sampling statistics"""
        return {
            'current_sampling_weights': dict(self.current_sampling_weights),
            'excluded_classes': dict(self.excluded_classes),
            'adaptation_rate': self.adaptation_rate
        }


class CurriculumProgressMonitor:
    """
    Monitors and visualizes curriculum learning progress.
    """
    
    def __init__(self, save_dir: str = "curriculum_logs"):
        """
        Initialize progress monitor.
        
        Args:
            save_dir: Directory to save logs and visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Tracking data
        self.epoch_logs = []
        self.stage_transitions = []
        self.performance_trends = defaultdict(list)
        
        logger.info(f"CurriculumProgressMonitor initialized, saving to {save_dir}")
    
    def log_epoch(self, 
                  epoch: int,
                  curriculum_stats: Dict,
                  performance_metrics: Dict,
                  loss_weights: Dict,
                  sampling_weights: Dict):
        """Log information for current epoch"""
        epoch_log = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'curriculum_stats': curriculum_stats,
            'performance_metrics': performance_metrics,
            'loss_weights': loss_weights,
            'sampling_weights': sampling_weights
        }
        
        self.epoch_logs.append(epoch_log)
        
        # Track stage transitions
        if (len(self.epoch_logs) > 1 and 
            curriculum_stats['current_stage'] != self.epoch_logs[-2]['curriculum_stats']['current_stage']):
            
            transition = {
                'epoch': epoch,
                'from_stage': self.epoch_logs[-2]['curriculum_stats']['current_stage'],
                'to_stage': curriculum_stats['current_stage'],
                'from_stage_name': self.epoch_logs[-2]['curriculum_stats']['current_stage_name'],
                'to_stage_name': curriculum_stats['current_stage_name']
            }
            self.stage_transitions.append(transition)
            
            logger.info(f"Stage transition at epoch {epoch}: "
                       f"{transition['from_stage_name']} -> {transition['to_stage_name']}")
        
        # Update performance trends
        for metric_name, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                self.performance_trends[metric_name].append((epoch, value))
    
    def generate_progress_visualization(self, save_path: str = None) -> str:
        """
        Generate comprehensive progress visualization.
        
        Args:
            save_path: Path to save visualization (optional)
            
        Returns:
            Path to saved visualization
        """
        if not self.epoch_logs:
            logger.warning("No epoch logs available for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Curriculum Learning Progress', fontsize=16)
        
        # Extract data for plotting
        epochs = [log['epoch'] for log in self.epoch_logs]
        stages = [log['curriculum_stats']['current_stage'] for log in self.epoch_logs]
        stage_progress = [log['curriculum_stats']['stage_progress'] for log in self.epoch_logs]
        overall_progress = [log['curriculum_stats']['overall_progress'] for log in self.epoch_logs]
        
        # Plot 1: Stage progression
        ax1 = axes[0, 0]
        ax1.plot(epochs, stages, 'b-', linewidth=2, label='Current Stage')
        ax1.fill_between(epochs, stages, alpha=0.3)
        
        # Mark stage transitions
        for transition in self.stage_transitions:
            ax1.axvline(x=transition['epoch'], color='red', linestyle='--', alpha=0.7)
            ax1.text(transition['epoch'], transition['to_stage'], 
                    f"→{transition['to_stage_name']}", rotation=90, fontsize=8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Curriculum Stage')
        ax1.set_title('Curriculum Stage Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Progress within stages
        ax2 = axes[0, 1]
        ax2.plot(epochs, stage_progress, 'g-', linewidth=2, label='Stage Progress')
        ax2.plot(epochs, overall_progress, 'orange', linewidth=2, label='Overall Progress')
        ax2.fill_between(epochs, stage_progress, alpha=0.3, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Progress (0-1)')
        ax2.set_title('Learning Progress')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Performance metrics
        ax3 = axes[1, 0]
        
        # Plot key performance metrics
        key_metrics = ['combined_macro_recall', 'action_macro_recall', 'severity_macro_recall']
        colors = ['blue', 'green', 'red']
        
        for metric, color in zip(key_metrics, colors):
            if metric in self.performance_trends:
                metric_epochs, metric_values = zip(*self.performance_trends[metric])
                ax3.plot(metric_epochs, metric_values, color=color, linewidth=2, 
                        label=metric.replace('_', ' ').title())
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall')
        ax3.set_title('Performance Metrics Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Minority class performance
        ax4 = axes[1, 1]
        
        # Extract minority class performance if available
        minority_metrics = []
        for log in self.epoch_logs:
            metrics = log['performance_metrics']
            epoch = log['epoch']
            
            # Action minority classes (Pushing=4, Dive=7)
            if 'action_class_recall' in metrics:
                action_recalls = metrics['action_class_recall']
                if len(action_recalls) > 7:
                    pushing_recall = action_recalls[4]
                    dive_recall = action_recalls[7]
                    minority_metrics.append((epoch, 'Pushing', pushing_recall))
                    minority_metrics.append((epoch, 'Dive', dive_recall))
            
            # Severity minority class (Red Card=3)
            if 'severity_class_recall' in metrics:
                severity_recalls = metrics['severity_class_recall']
                if len(severity_recalls) > 3:
                    red_card_recall = severity_recalls[3]
                    minority_metrics.append((epoch, 'Red Card', red_card_recall))
        
        # Plot minority class performance
        if minority_metrics:
            minority_data = defaultdict(list)
            for epoch, class_name, recall in minority_metrics:
                minority_data[class_name].append((epoch, recall))
            
            colors = ['red', 'blue', 'green']
            for i, (class_name, data) in enumerate(minority_data.items()):
                if data:
                    epochs_data, recalls_data = zip(*data)
                    ax4.plot(epochs_data, recalls_data, color=colors[i % len(colors)], 
                            linewidth=2, label=class_name)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.set_title('Minority Class Performance')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"curriculum_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Curriculum progress visualization saved to {save_path}")
        return save_path
    
    def save_progress_logs(self, save_path: str = None) -> str:
        """Save detailed progress logs to JSON file"""
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"curriculum_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logs_data = {
            'epoch_logs': self.epoch_logs,
            'stage_transitions': self.stage_transitions,
            'performance_trends': {k: list(v) for k, v in self.performance_trends.items()},
            'summary': {
                'total_epochs_logged': len(self.epoch_logs),
                'total_stage_transitions': len(self.stage_transitions),
                'metrics_tracked': list(self.performance_trends.keys())
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        logger.info(f"Curriculum progress logs saved to {save_path}")
        return save_path
    
    def get_summary_report(self) -> Dict:
        """Generate summary report of curriculum learning progress"""
        if not self.epoch_logs:
            return {'error': 'No data available'}
        
        # Calculate summary statistics
        final_log = self.epoch_logs[-1]
        final_metrics = final_log['performance_metrics']
        
        # Performance improvement analysis
        if len(self.epoch_logs) > 1:
            initial_metrics = self.epoch_logs[0]['performance_metrics']
            
            improvements = {}
            for metric in ['combined_macro_recall', 'action_macro_recall', 'severity_macro_recall']:
                if metric in final_metrics and metric in initial_metrics:
                    improvement = final_metrics[metric] - initial_metrics[metric]
                    improvements[metric] = {
                        'initial': initial_metrics[metric],
                        'final': final_metrics[metric],
                        'improvement': improvement,
                        'improvement_percent': (improvement / max(initial_metrics[metric], 0.001)) * 100
                    }
        else:
            improvements = {}
        
        # Stage analysis
        stage_durations = []
        if len(self.stage_transitions) > 0:
            # Calculate duration of each completed stage
            for i in range(len(self.stage_transitions)):
                if i == 0:
                    start_epoch = 0
                else:
                    start_epoch = self.stage_transitions[i-1]['epoch']
                
                end_epoch = self.stage_transitions[i]['epoch']
                duration = end_epoch - start_epoch
                stage_durations.append({
                    'stage': self.stage_transitions[i]['from_stage'],
                    'stage_name': self.stage_transitions[i]['from_stage_name'],
                    'duration': duration
                })
        
        summary = {
            'total_epochs': len(self.epoch_logs),
            'final_stage': final_log['curriculum_stats']['current_stage'],
            'final_stage_name': final_log['curriculum_stats']['current_stage_name'],
            'overall_progress': final_log['curriculum_stats']['overall_progress'],
            'stage_transitions': len(self.stage_transitions),
            'performance_improvements': improvements,
            'stage_durations': stage_durations,
            'final_performance': {
                'combined_macro_recall': final_metrics.get('combined_macro_recall', 0),
                'action_macro_recall': final_metrics.get('action_macro_recall', 0),
                'severity_macro_recall': final_metrics.get('severity_macro_recall', 0)
            }
        }
        
        return summary


class CurriculumLearningManager:
    """
    Main manager class that coordinates all curriculum learning components.
    """
    
    def __init__(self,
                 total_epochs: int,
                 base_sampler = None,
                 config: Dict = None,
                 save_dir: str = "curriculum_learning_logs"):
        """
        Initialize curriculum learning manager.
        
        Args:
            total_epochs: Total number of training epochs
            base_sampler: Base sampler to enhance with curriculum awareness
            config: Configuration dictionary
            save_dir: Directory to save logs and outputs
        """
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Use default config if none provided
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Initialize curriculum scheduler
        self.curriculum_scheduler = CurriculumScheduler(
            total_epochs=total_epochs,
            **config.get('scheduler', {})
        )
        
        # Initialize curriculum-aware sampler if base sampler provided
        self.curriculum_sampler = None
        if base_sampler is not None:
            self.curriculum_sampler = CurriculumAwareSampler(
                base_sampler=base_sampler,
                curriculum_scheduler=self.curriculum_scheduler,
                **config.get('sampler', {})
            )
        
        # Initialize progress monitor
        self.progress_monitor = CurriculumProgressMonitor(
            save_dir=os.path.join(save_dir, 'progress')
        )
        
        # Current epoch tracking
        self.current_epoch = 0
        
        logger.info(f"CurriculumLearningManager initialized:")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Save directory: {save_dir}")
        logger.info(f"  Curriculum sampler: {'Enabled' if self.curriculum_sampler else 'Disabled'}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for curriculum learning"""
        return {
            'scheduler': {
                'action_class_difficulty': {
                    0: 'easy', 1: 'easy', 2: 'medium', 3: 'medium',
                    4: 'hard', 5: 'medium', 6: 'medium', 7: 'hard'
                },
                'severity_class_difficulty': {
                    0: 'easy', 1: 'easy', 2: 'medium', 3: 'hard'
                },
                'stage_ratios': (0.4, 0.35, 0.25),  # Easy, Medium, Hard
                'smooth_transitions': True,
                'transition_overlap': 3
            },
            'sampler': {
                'adaptation_rate': 0.15
            }
        }
    
    def get_epoch_configuration(self, epoch: int) -> Dict:
        """
        Get complete configuration for current epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Dictionary with curriculum configuration
        """
        self.current_epoch = epoch
        
        # Get curriculum stage configuration
        current_stage, stage_config = self.curriculum_scheduler.get_current_stage_config(epoch)
        
        # Get curriculum statistics
        curriculum_stats = self.curriculum_scheduler.get_curriculum_stats()
        
        # Get sampler statistics if available
        sampler_stats = {}
        if self.curriculum_sampler:
            self.curriculum_sampler.update_for_epoch(epoch)
            sampler_stats = self.curriculum_sampler.get_sampling_stats()
        
        # Combine all configuration
        epoch_config = {
            'epoch': epoch,
            'current_stage': current_stage,
            'stage_config': stage_config,
            'curriculum_stats': curriculum_stats,
            'sampler_stats': sampler_stats,
            'loss_weights': stage_config['loss_weights'],
            'sampling_weights': stage_config['sampling_weights'],
            'included_classes': stage_config['included_classes']
        }
        
        return epoch_config
    
    def update_performance_metrics(self, epoch: int, metrics: Dict):
        """
        Update performance metrics for curriculum tracking.
        
        Args:
            epoch: Current epoch
            metrics: Performance metrics dictionary
        """
        # Update curriculum scheduler
        self.curriculum_scheduler.update_performance_history(epoch, metrics)
        
        # Get current configuration for logging
        epoch_config = self.get_epoch_configuration(epoch)
        
        # Log to progress monitor
        self.progress_monitor.log_epoch(
            epoch=epoch,
            curriculum_stats=epoch_config['curriculum_stats'],
            performance_metrics=metrics,
            loss_weights=epoch_config['loss_weights'],
            sampling_weights=epoch_config['sampling_weights']
        )
    
    def get_batch_indices(self, epoch: int, batch_size: int) -> List[int]:
        """
        Get batch indices respecting curriculum constraints.
        
        Args:
            epoch: Current epoch
            batch_size: Desired batch size
            
        Returns:
            List of sample indices
        """
        if self.curriculum_sampler:
            return self.curriculum_sampler.get_batch_indices(epoch, batch_size)
        else:
            logger.warning("No curriculum sampler available, returning empty list")
            return []
    
    def generate_progress_report(self) -> Tuple[str, str, Dict]:
        """
        Generate comprehensive progress report.
        
        Returns:
            Tuple of (visualization_path, logs_path, summary_report)
        """
        logger.info("Generating curriculum learning progress report...")
        
        # Generate visualization
        viz_path = self.progress_monitor.generate_progress_visualization()
        
        # Save detailed logs
        logs_path = self.progress_monitor.save_progress_logs()
        
        # Generate summary report
        summary_report = self.progress_monitor.get_summary_report()
        
        # Save summary report
        summary_path = os.path.join(self.save_dir, 'curriculum_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Progress report generated:")
        logger.info(f"  Visualization: {viz_path}")
        logger.info(f"  Detailed logs: {logs_path}")
        logger.info(f"  Summary: {summary_path}")
        
        return viz_path, logs_path, summary_report
    
    def save_checkpoint(self, epoch: int, additional_data: Dict = None) -> str:
        """
        Save curriculum learning checkpoint.
        
        Args:
            epoch: Current epoch
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'epoch': epoch,
            'curriculum_scheduler_state': {
                'progress': self.curriculum_scheduler.progress.__dict__,
                'stages': [stage.__dict__ for stage in self.curriculum_scheduler.stages],
                'config': self.config
            },
            'current_configuration': self.get_epoch_configuration(epoch),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            checkpoint_data['additional_data'] = additional_data
        
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"curriculum_checkpoint_epoch_{epoch}.json"
        )
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Curriculum checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics about curriculum learning"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'curriculum_stats': self.curriculum_scheduler.get_curriculum_stats(),
            'sampler_stats': self.curriculum_sampler.get_sampling_stats() if self.curriculum_sampler else {},
            'progress_summary': self.progress_monitor.get_summary_report(),
            'config': self.config
        }


def create_curriculum_learning_setup(total_epochs: int,
                                    base_sampler = None,
                                    config: Dict = None) -> CurriculumLearningManager:
    """
    Create a complete curriculum learning setup.
    
    Args:
        total_epochs: Total number of training epochs
        base_sampler: Base sampler to enhance (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Configured CurriculumLearningManager
    """
    logger.info("Creating curriculum learning setup...")
    
    manager = CurriculumLearningManager(
        total_epochs=total_epochs,
        base_sampler=base_sampler,
        config=config
    )
    
    logger.info("Curriculum learning setup complete!")
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Test curriculum learning system
    logger.info("Testing Curriculum Learning System...")
    
    # Create test setup
    total_epochs = 25
    manager = create_curriculum_learning_setup(total_epochs)
    
    # Simulate training epochs
    for epoch in range(total_epochs):
        # Get epoch configuration
        config = manager.get_epoch_configuration(epoch)
        
        logger.info(f"Epoch {epoch}: Stage '{config['current_stage'].stage_name}' "
                   f"(Progress: {config['curriculum_stats']['stage_progress']:.2f})")
        
        # Simulate performance metrics
        fake_metrics = {
            'combined_macro_recall': 0.2 + 0.3 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'action_macro_recall': 0.25 + 0.25 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'severity_macro_recall': 0.15 + 0.35 * (epoch / total_epochs) + np.random.normal(0, 0.02),
            'action_class_recall': [0.8, 0.7, 0.6, 0.5, 0.1 + 0.2 * (epoch / total_epochs), 0.6, 0.5, 0.05 + 0.15 * (epoch / total_epochs)],
            'severity_class_recall': [0.9, 0.8, 0.6, 0.05 + 0.25 * (epoch / total_epochs)]
        }
        
        # Update performance metrics
        manager.update_performance_metrics(epoch, fake_metrics)
    
    # Generate final report
    viz_path, logs_path, summary = manager.generate_progress_report()
    
    logger.info("Curriculum learning test completed!")
    logger.info(f"Final performance: {summary.get('final_performance', {})}")