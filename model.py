import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch.nn as nn
from dataset import MVFoulsDataset, custom_collate_fn
from torch.utils.data import DataLoader
import os

class MultiClipAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

        # Final linear layer for refined aggregation after multi-head attention
        self.out_transform = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x will be (num_clips, embed_dim) initially
        is_single_input = False
        if x.dim() == 2:
            # If input is (num_clips, embed_dim), unsqueeze to (1, num_clips, embed_dim)
            x = x.unsqueeze(0)  # (batch_size=1, seq_len=num_clips, embed_dim)
            is_single_input = True
        
        batch_size, seq_len, embed_dim = x.size()

        # Apply linear transformations and split into heads
        queries = self.query_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5) # Scale by sqrt(d_k)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        weighted_values = torch.matmul(attention_weights, values)

        # Concatenate heads and apply final linear layer
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_dim)
        concat_heads = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Aggregate features: sum along the sequence length (clips) dimension
        aggregated_features = torch.sum(concat_heads, dim=1) # (batch_size, embed_dim)
        
        # Apply final output transformation
        aggregated_features = self.out_transform(aggregated_features)

        if is_single_input:
            return aggregated_features.squeeze(0) # Remove batch dimension if it was added
        
        return aggregated_features

class MVFoulsModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=weights, progress=True)

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
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(0.3)
        )

        # Initialize the MultiClipAttention module with num_heads
        self.attention_module = MultiClipAttention(in_features, num_heads=4) # Using 4 heads as an example

        # Define new classification heads for action and severity, connected to the shared head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8)  # 8 classes for action
        )
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4) # 4 classes for severity
        )

    def forward(self, x):
        # x is a batch of videos from the DataLoader, shape: (Total_Clips_in_Batch, C, T, H, W)
        
        # The MViT model expects input in (B, C, T, H, W) format, which is already provided by the DataLoader
        all_clips_batch = x
        print(f"Shape of all_clips_batch before MViT model: {all_clips_batch.shape}")
        # Pass through the MViT backbone
        clip_features = self.model(all_clips_batch) # (total_num_clips, in_features)

        # Apply attention mechanism to combine features from multiple clips if necessary
        # If there's only one clip per action in the batch, this might not be needed or will act as passthrough
        # For now, assuming clip_features is already the combined features if needed, or single features.
        
        # Apply shared head
        shared_features = self.shared_head(clip_features)

        # Apply action and severity heads
        action_logits = self.action_head(shared_features)
        severity_logits = self.severity_head(shared_features)

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

    # --- Test MultiClipAttention independently ---
    print("\nTesting MultiClipAttention independently...")
    embed_dim = 768 # MViT_V2_S_Weights.KINETICS400_V1 features dimension
    num_heads = 4
    num_clips_test = 5 # Example: 5 clips for one action
    dummy_clip_features = torch.randn(num_clips_test, embed_dim) # (num_clips, embed_dim)

    try:
        attention_module_test = MultiClipAttention(embed_dim, num_heads=num_heads)
        aggregated_output = attention_module_test(dummy_clip_features)
        print(f"  MultiClipAttention input shape: {dummy_clip_features.shape}")
        print(f"  MultiClipAttention output shape: {aggregated_output.shape}")
        expected_output_shape = torch.Size([embed_dim])
        if aggregated_output.shape == expected_output_shape:
            print("  MultiClipAttention output shape test PASSED.")
        else:
            print(f"  MultiClipAttention output shape test FAILED. Expected {expected_output_shape}, got {aggregated_output.shape}")
    except Exception as e:
        print(f"  Error during MultiClipAttention independent test: {e}")

    # --- Test with real data from dataset ---
    print("\nTesting with real data from MVFoulsDataset...")
    test_folder = "mvfouls"  
    test_split = "train"
    start_frame = 67
    end_frame = 82

    if not os.path.exists(test_folder):
        print(f"Error: Dataset folder '{test_folder}' not found. Please ensure your dataset is downloaded and extracted.")
        exit()

    try:
        dataset = MVFoulsDataset(test_folder, test_split, start_frame, end_frame)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
        print(f"Dataset initialized with {len(dataset)} samples. DataLoader created.")
    except Exception as e:
        print(f"Error initializing dataset or dataloader: {e}")
        exit()

    if len(dataset) == 0:
        print("No samples found in the dataset. Exiting test.")
        exit()

    # Get a real batch
    try:
        real_batch_videos, real_batch_action_labels, real_batch_severity_labels, _ = next(iter(dataloader)) # Added _ to unpack action_ids
        print(f"Got a real batch from DataLoader. Batch size: {len(real_batch_videos)}")
        for i, video_tensor in enumerate(real_batch_videos):
            print(f"  Video {i} shape: {video_tensor.shape}")
        print(f"  Action labels shape: {real_batch_action_labels.shape}")
        print(f"  Severity labels shape: {real_batch_severity_labels.shape}")

    except Exception as e:
        print(f"Error getting batch from DataLoader: {e}")
        exit()

    # Pass real batch through the model
    try:
        action_logits, severity_logits = model(real_batch_videos)
        print("Model forward pass with real data successful.")
    except Exception as e:
        print(f"Error during forward pass with real data: {e}")
        exit()

    # Assert output shapes using the actual batch size
    actual_batch_size = len(real_batch_videos)
    expected_action_shape = torch.Size([actual_batch_size, 8])
    expected_severity_shape = torch.Size([actual_batch_size, 4])

    if action_logits.shape == expected_action_shape:
        print(f"Action logits shape check passed: {action_logits.shape}")
    else:
        print(f"Action logits shape check FAILED. Expected {expected_action_shape}, got {action_logits.shape}")

    if severity_logits.shape == expected_severity_shape:
        print(f"Severity logits shape check passed: {severity_logits.shape}")
    else:
        print(f"Severity logits shape check FAILED. Expected {expected_severity_shape}, got {severity_logits.shape}")

    if action_logits.shape == expected_action_shape and severity_logits.shape == expected_severity_shape:
        print("All tests passed with real data!")
    else:
        print("Some tests failed with real data.")
