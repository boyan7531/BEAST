import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch.nn as nn

class MultiClipAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x will be (num_clips, embed_dim)
        # Apply linear transformations to get Q, K, V
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)

        # Calculate attention scores
        # (num_clips, embed_dim) @ (embed_dim, num_clips) -> (num_clips, num_clips)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (keys.size(-1) ** 0.5) # Scale by sqrt(d_k)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        # (num_clips, num_clips) @ (num_clips, embed_dim) -> (num_clips, embed_dim)
        weighted_values = torch.matmul(attention_weights, values)

        # For multi-clip attention, we want to aggregate these. A simple sum or mean is common.
        # Here, we'll sum them up to get a single feature vector for the action.
        aggregated_features = torch.sum(weighted_values, dim=0) # (embed_dim)
        return aggregated_features

class MVFoulsModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=weights)

        # Get the number of input features from the backbone.
        # 1 because 0 is the dropout layer
        in_features = self.model.head[1].in_features

        # Remove the original classification head as we're replacing it.
        self.model.head = nn.Identity()

        # Define a shared custom head for feature processing
        hidden_dim = 512
        self.shared_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Initialize the MultiClipAttention module
        self.attention_module = MultiClipAttention(in_features)

        # Define new classification heads for action and severity, connected to the shared head
        self.action_head = nn.Linear(hidden_dim, 8)  # 8 classes for action
        self.severity_head = nn.Linear(hidden_dim, 4) # 4 classes for severity

    def forward(self, x: list[torch.Tensor]):
        # x is now expected to be a list of tensors, where each tensor is
        # (num_clips_for_this_action, C, num_frames, H, W)
        
        # Store original batch sizes (number of clips per action)
        num_clips_per_action_batch = [item.size(0) for item in x]

        # Concatenate all clips from the batch into a single tensor for backbone processing
        # This allows processing all clips efficiently in one go
        all_clips_batch = torch.cat(x, dim=0)

        # Get the features from the backbone model for each individual clip
        clip_features = self.model(all_clips_batch) # (total_num_clips, in_features)

        # Split the features back into individual action features based on original clip counts
        # This results in a list of tensors, where each tensor is (num_clips_for_this_action, in_features)
        split_clip_features = torch.split(clip_features, num_clips_per_action_batch)

        # Apply attention for each action in the batch
        aggregated_features_batch = []
        for single_action_clip_features in split_clip_features:
            # attention_module expects (num_clips, embed_dim)
            aggregated_feature = self.attention_module(single_action_clip_features) # (in_features)
            aggregated_features_batch.append(aggregated_feature)
        
        # Stack the aggregated features back into a batch tensor
        processed_features = torch.stack(aggregated_features_batch) # (batch_size, in_features)

        # Pass features through the shared custom head
        processed_features = self.shared_head(processed_features)

        # Pass the processed features through the action and severity heads
        action_logits = self.action_head(processed_features)
        severity_logits = self.severity_head(processed_features)

        # Return both sets of logits
        return action_logits, severity_logits
        

if __name__ == "__main__":
    print("Running tests for MVFoulsModel...")

    # Instantiate the model
    try:
        model = MVFoulsModel()
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        exit()
        
    print("\nParameter counts for new heads:")
    shared_head_params = sum(p.numel() for p in model.shared_head.parameters() if p.requires_grad)
    print(f"Shared head parameters: {shared_head_params}")
    action_head_params = sum(p.numel() for p in model.action_head.parameters() if p.requires_grad)
    print(f"Action head parameters: {action_head_params}")

    severity_head_params = sum(p.numel() for p in model.severity_head.parameters() if p.requires_grad)
    print(f"Severity head parameters: {severity_head_params}")

    # Create a dummy input tensor
    # MViT_V2_S typically expects input in the format [batch_size, channels, frames, height, width]
    # For a video model, common input shape might be (batch_size, 3, num_frames, 224, 224)
    batch_size = 2
    num_clips_per_action = 3 # New: Number of clips per action
    num_frames = 16 # Example number of frames per clip
    height = 224
    width = 224
    # Update dummy input shape to include num_clips_per_action
    # dummy_input = torch.randn(batch_size, num_clips_per_action, 3, num_frames, height, width)

    # Create a list of dummy inputs with varying number of clips per action
    dummy_input = []
    for i in range(batch_size):
        # Varying num_clips_per_action for each item in the batch
        current_num_clips = 2 if i % 2 == 0 else 4 # Example: 2 clips for even batch items, 4 for odd
        dummy_input.append(torch.randn(current_num_clips, 3, num_frames, height, width))
    
    print(f"Created dummy input as a list of tensors. Example shapes: {[item.shape for item in dummy_input]}")

    # Pass dummy input through the model
    try:
        action_logits, severity_logits = model(dummy_input)
        print("Model forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        exit()

    # Assert output shapes
    # The batch_size here refers to the number of actions in the batch, not total clips
    expected_action_shape = torch.Size([batch_size, 8])
    expected_severity_shape = torch.Size([batch_size, 4])

    if action_logits.shape == expected_action_shape:
        print(f"Action logits shape check passed: {action_logits.shape}")
    else:
        print(f"Action logits shape check FAILED. Expected {expected_action_shape}, got {action_logits.shape}")

    if severity_logits.shape == expected_severity_shape:
        print(f"Severity logits shape check passed: {severity_logits.shape}")
    else:
        print(f"Severity logits shape check FAILED. Expected {expected_severity_shape}, got {severity_logits.shape}")

    if action_logits.shape == expected_action_shape and severity_logits.shape == expected_severity_shape:
        print("All tests passed!")
    else:
        print("Some tests failed.")
