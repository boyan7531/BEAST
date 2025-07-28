import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import math
from typing import Tuple, Dict, List


class ClassSpecificAttention(nn.Module):
    """
    Class-specific attention mechanism for action and severity tasks.
    Implements requirement 5.1: class-specific attention mechanisms.
    """
    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Class-specific query generators
        self.class_queries = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # Shared key and value transformations
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.class_queries)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        Returns:
            attended_features: (batch_size, num_classes, embed_dim)
            attention_weights: (batch_size, num_classes, seq_len)
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len, embed_dim = features.shape
        
        # Generate keys and values
        keys = self.key_transform(features)  # (batch_size, seq_len, embed_dim)
        values = self.value_transform(features)  # (batch_size, seq_len, embed_dim)
        
        # Expand class queries for batch
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_classes, embed_dim)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, num_classes, seq_len)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)  # (batch_size, num_heads, num_classes, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.num_classes, embed_dim)
        
        # Apply output projection
        attended_features = self.out_proj(attended)
        
        # Average attention weights across heads for interpretability
        avg_attention_weights = attention_weights.mean(dim=1)  # (batch_size, num_classes, seq_len)
        
        if squeeze_output:
            attended_features = attended_features.squeeze(0)
            avg_attention_weights = avg_attention_weights.squeeze(0)
            
        return attended_features, avg_attention_weights


class TaskSpecificFeatureExtractor(nn.Module):
    """
    Task-specific feature extractor with shared backbone and cross-task sharing.
    Implements requirement 5.2: separate feature extractors for action and severity tasks.
    """
    def __init__(self, shared_dim: int, task_specific_dim: int, num_action_classes: int, num_severity_classes: int):
        super().__init__()
        self.shared_dim = shared_dim
        self.task_specific_dim = task_specific_dim
        
        # Shared feature processing
        self.shared_processor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.LayerNorm(shared_dim),
            nn.Dropout(0.1)
        )
        
        # Task-specific processors
        self.action_processor = nn.Sequential(
            nn.Linear(shared_dim, task_specific_dim),
            nn.GELU(),
            nn.LayerNorm(task_specific_dim),
            nn.Dropout(0.2)
        )
        
        self.severity_processor = nn.Sequential(
            nn.Linear(shared_dim, task_specific_dim),
            nn.GELU(),
            nn.LayerNorm(task_specific_dim),
            nn.Dropout(0.2)
        )
        
        # Cross-task interaction layers
        self.action_to_severity = nn.Linear(task_specific_dim, task_specific_dim // 2)
        self.severity_to_action = nn.Linear(task_specific_dim, task_specific_dim // 2)
        
        # Class-specific attention modules (for enhanced features with cross-task info)
        enhanced_dim = task_specific_dim + task_specific_dim // 2
        self.action_attention = ClassSpecificAttention(enhanced_dim, num_action_classes)
        self.severity_attention = ClassSpecificAttention(enhanced_dim, num_severity_classes)
        
    def forward(self, shared_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            shared_features: (batch_size, seq_len, shared_dim) or (seq_len, shared_dim)
        Returns:
            action_features: (batch_size, num_action_classes, task_specific_dim)
            severity_features: (batch_size, num_severity_classes, task_specific_dim)
            attention_weights: Dict with attention weights for interpretability
        """
        # Process shared features
        processed_shared = self.shared_processor(shared_features)
        
        # Generate task-specific features
        action_features = self.action_processor(processed_shared)
        severity_features = self.severity_processor(processed_shared)
        
        # Cross-task interaction
        action_to_sev = self.action_to_severity(action_features)
        sev_to_action = self.severity_to_action(severity_features)
        
        # Enhance features with cross-task information
        enhanced_action = torch.cat([action_features, sev_to_action], dim=-1)
        enhanced_severity = torch.cat([severity_features, action_to_sev], dim=-1)
        
        # Apply class-specific attention
        action_attended, action_attn_weights = self.action_attention(enhanced_action)
        severity_attended, severity_attn_weights = self.severity_attention(enhanced_severity)
        
        attention_weights = {
            'action_attention': action_attn_weights,
            'severity_attention': severity_attn_weights
        }
        
        return action_attended, severity_attended, attention_weights


class ConfidenceEstimator(nn.Module):
    """
    Confidence estimation module with uncertainty quantification.
    Implements requirement 5.4: confidence-based ensemble methods.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
        
        # Uncertainty estimation (aleatoric and epistemic)
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Softplus()  # Ensure positive values
        )
        
    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, num_classes, input_dim)
            logits: (batch_size, num_classes)
        Returns:
            confidence: (batch_size, num_classes) - confidence scores
            aleatoric_uncertainty: (batch_size, num_classes) - data uncertainty
            epistemic_uncertainty: (batch_size, num_classes) - model uncertainty
        """
        # Compute confidence for each class prediction
        if features.dim() == 3:
            batch_size, num_classes, input_dim = features.shape
            confidence = self.confidence_net(features.view(-1, input_dim)).view(batch_size, num_classes)
            aleatoric_uncertainty = self.aleatoric_net(features.view(-1, input_dim)).view(batch_size, num_classes)
        else:
            confidence = self.confidence_net(features)
            aleatoric_uncertainty = self.aleatoric_net(features)
        
        # Epistemic uncertainty from prediction entropy
        probs = F.softmax(logits, dim=-1)
        epistemic_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        
        if epistemic_uncertainty.shape != confidence.shape:
            epistemic_uncertainty = epistemic_uncertainty.expand_as(confidence)
        
        return confidence, aleatoric_uncertainty, epistemic_uncertainty


class WeightedMultiClipAttention(nn.Module):
    """
    Weighted multi-clip attention based on clip importance.
    Implements requirement 5.3: weighted attention based on clip importance.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Standard attention components
        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)
        
        # Clip importance estimation
        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_transform = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (num_clips, embed_dim) or (batch_size, num_clips, embed_dim)
        Returns:
            aggregated_features: (embed_dim) or (batch_size, embed_dim)
            clip_importance_weights: (num_clips) or (batch_size, num_clips)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len, embed_dim = x.shape
        
        # Estimate clip importance
        clip_importance = self.importance_net(x).squeeze(-1)  # (batch_size, seq_len)
        importance_weights = F.softmax(clip_importance, dim=-1)
        
        # Standard multi-head attention
        queries = self.query_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Concatenate heads
        concat_heads = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply importance weighting
        importance_weights_expanded = importance_weights.unsqueeze(-1).expand_as(concat_heads)
        weighted_features = concat_heads * importance_weights_expanded
        
        # Aggregate with importance weights
        aggregated_features = torch.sum(weighted_features, dim=1)  # (batch_size, embed_dim)
        
        # Apply output transformation
        aggregated_features = self.out_transform(aggregated_features)
        
        if squeeze_output:
            aggregated_features = aggregated_features.squeeze(0)
            importance_weights = importance_weights.squeeze(0)
            
        return aggregated_features, importance_weights


class AttentionDiversityLoss(nn.Module):
    """
    Attention diversity loss to prevent attention collapse.
    Encourages different attention heads to focus on different aspects.
    """
    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
        
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights: (batch_size, num_heads, seq_len, seq_len) or similar
        Returns:
            diversity_loss: scalar tensor
        """
        if attention_weights.dim() < 3:
            return torch.tensor(0.0, device=attention_weights.device)
            
        # Compute pairwise cosine similarity between attention heads
        batch_size = attention_weights.shape[0]
        num_heads = attention_weights.shape[1]
        
        # Flatten attention patterns for each head
        flattened_attention = attention_weights.view(batch_size, num_heads, -1)
        
        # Normalize attention patterns
        normalized_attention = F.normalize(flattened_attention, p=2, dim=-1)
        
        # Compute pairwise cosine similarities
        similarities = torch.matmul(normalized_attention, normalized_attention.transpose(-2, -1))
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(num_heads, device=similarities.device).bool()
        similarities = similarities.masked_fill(mask.unsqueeze(0), 0)
        
        # Compute diversity loss (penalize high similarities)
        diversity_loss = similarities.abs().mean() * self.diversity_weight
        
        return diversity_loss


class EnhancedMVFoulsModel(nn.Module):
    """
    Enhanced MVFouls model with class-specific components, confidence estimation,
    and attention diversity mechanisms.
    """
    def __init__(self, num_action_classes: int = 8, num_severity_classes: int = 4):
        super().__init__()
        
        # Load pre-trained MViT backbone
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.backbone = mvit_v2_s(weights=weights, progress=True)
        
        # Get backbone output dimension
        backbone_dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()  # Remove original head
        
        # Model dimensions
        self.backbone_dim = backbone_dim
        self.shared_dim = 512
        self.task_specific_dim = 256
        self.num_action_classes = num_action_classes
        self.num_severity_classes = num_severity_classes
        
        # Weighted multi-clip attention
        self.clip_attention = WeightedMultiClipAttention(backbone_dim)
        
        # Task-specific feature extractor
        self.feature_extractor = TaskSpecificFeatureExtractor(
            shared_dim=self.shared_dim,
            task_specific_dim=self.task_specific_dim,
            num_action_classes=num_action_classes,
            num_severity_classes=num_severity_classes
        )
        
        # Shared feature projection
        self.shared_projection = nn.Sequential(
            nn.Linear(backbone_dim, self.shared_dim),
            nn.GELU(),
            nn.LayerNorm(self.shared_dim),
            nn.Dropout(0.1)
        )
        
        # Classification heads
        self.action_classifier = nn.Sequential(
            nn.Linear(self.task_specific_dim + self.task_specific_dim // 2, self.task_specific_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.task_specific_dim, num_action_classes)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(self.task_specific_dim + self.task_specific_dim // 2, self.task_specific_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.task_specific_dim, num_severity_classes)
        )
        
        # Confidence estimators
        self.action_confidence = ConfidenceEstimator(
            self.task_specific_dim + self.task_specific_dim // 2, 
            num_action_classes
        )
        self.severity_confidence = ConfidenceEstimator(
            self.task_specific_dim + self.task_specific_dim // 2, 
            num_severity_classes
        )
        
        # Attention diversity loss
        self.diversity_loss = AttentionDiversityLoss()
        
    def forward(self, x_list: List[torch.Tensor], return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x_list: List of tensors, each with shape (num_clips, C, T, H, W)
            return_attention: Whether to return attention weights
        Returns:
            action_logits: (batch_size, num_action_classes)
            severity_logits: (batch_size, num_severity_classes)
            confidence_scores: Dict with confidence and uncertainty estimates
            attention_info: Dict with attention weights and diversity loss
        """
        batch_features = []
        clip_importance_weights = []
        
        # Process each sample in the batch
        for x_clips in x_list:
            # Extract features from all clips
            clip_features = self.backbone(x_clips)  # (num_clips, backbone_dim)
            
            # Apply weighted multi-clip attention
            aggregated_features, importance_weights = self.clip_attention(clip_features)
            
            batch_features.append(aggregated_features)
            clip_importance_weights.append(importance_weights)
        
        # Stack batch features
        batch_features = torch.stack(batch_features, dim=0)  # (batch_size, backbone_dim)
        
        # Project to shared dimension
        shared_features = self.shared_projection(batch_features)  # (batch_size, shared_dim)
        
        # Extract task-specific features with class-specific attention
        # Need to process each sample in the batch separately for class-specific attention
        batch_action_features = []
        batch_severity_features = []
        batch_attention_weights = []
        
        for i in range(shared_features.shape[0]):
            sample_features = shared_features[i:i+1]  # Keep batch dimension
            action_feat, severity_feat, attn_weights = self.feature_extractor(sample_features)
            batch_action_features.append(action_feat)
            batch_severity_features.append(severity_feat)
            batch_attention_weights.append(attn_weights)
        
        # Stack the results
        action_features = torch.stack(batch_action_features, dim=0)  # (batch_size, num_action_classes, enhanced_dim)
        severity_features = torch.stack(batch_severity_features, dim=0)  # (batch_size, num_severity_classes, enhanced_dim)
        
        # Combine attention weights
        attention_weights = {
            'action_attention': torch.stack([aw['action_attention'] for aw in batch_attention_weights], dim=0),
            'severity_attention': torch.stack([aw['severity_attention'] for aw in batch_attention_weights], dim=0)
        }
        
        # Generate predictions
        # Generate predictions
        # action_features: (batch_size, num_action_classes, enhanced_dim)
        # severity_features: (batch_size, num_severity_classes, enhanced_dim)
        
        # Average over classes to get sample-level features
        action_pooled = action_features.mean(dim=1)  # (batch_size, enhanced_dim)
        severity_pooled = severity_features.mean(dim=1)  # (batch_size, enhanced_dim)
        
        # Generate final predictions
        action_logits = self.action_classifier(action_pooled)
        severity_logits = self.severity_classifier(severity_pooled)
        
        # Estimate confidence and uncertainty
        action_conf, action_aleatoric, action_epistemic = self.action_confidence(action_features, action_logits)
        severity_conf, severity_aleatoric, severity_epistemic = self.severity_confidence(severity_features, severity_logits)
        
        confidence_scores = {
            'action_confidence': action_conf.mean(dim=1),  # Average over classes
            'severity_confidence': severity_conf.mean(dim=1),
            'action_aleatoric_uncertainty': action_aleatoric.mean(dim=1),
            'severity_aleatoric_uncertainty': severity_aleatoric.mean(dim=1),
            'action_epistemic_uncertainty': action_epistemic.mean(dim=1),
            'severity_epistemic_uncertainty': severity_epistemic.mean(dim=1)
        }
        
        # Compute attention diversity loss
        diversity_loss = torch.tensor(0.0, device=action_logits.device)
        if 'action_attention' in attention_weights and attention_weights['action_attention'].dim() >= 3:
            diversity_loss += self.diversity_loss(attention_weights['action_attention'])
        if 'severity_attention' in attention_weights and attention_weights['severity_attention'].dim() >= 3:
            diversity_loss += self.diversity_loss(attention_weights['severity_attention'])
        
        attention_info = {
            'clip_importance_weights': clip_importance_weights,
            'class_attention_weights': attention_weights,
            'diversity_loss': diversity_loss
        }
        
        if not return_attention:
            attention_info = {'diversity_loss': diversity_loss}
        
        return action_logits, severity_logits, confidence_scores, attention_info
    
    def get_class_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return class-specific attention weights for interpretability."""
        return {
            'action_class_queries': self.feature_extractor.action_attention.class_queries,
            'severity_class_queries': self.feature_extractor.severity_attention.class_queries
        }


if __name__ == "__main__":
    print("Testing Enhanced MVFouls Model...")
    
    # Test model instantiation
    model = EnhancedMVFoulsModel()
    print("✓ Model instantiated successfully")
    
    # Test with dummy data
    batch_size = 2
    num_clips = 5
    dummy_videos = [
        torch.randn(num_clips, 3, 16, 224, 224) for _ in range(batch_size)
    ]
    
    print(f"Testing with batch size: {batch_size}, clips per sample: {num_clips}")
    
    try:
        action_logits, severity_logits, confidence_scores, attention_info = model(dummy_videos, return_attention=True)
        
        print(f"✓ Forward pass successful")
        print(f"  Action logits shape: {action_logits.shape}")
        print(f"  Severity logits shape: {severity_logits.shape}")
        print(f"  Confidence scores keys: {list(confidence_scores.keys())}")
        print(f"  Attention info keys: {list(attention_info.keys())}")
        print(f"  Diversity loss: {attention_info['diversity_loss'].item():.4f}")
        
        # Test confidence scores shapes
        for key, value in confidence_scores.items():
            print(f"  {key} shape: {value.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise
    
    print("\n✓ All tests passed!")