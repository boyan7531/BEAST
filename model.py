import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch.nn as nn

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
            nn.Dropout(0.3) # Add dropout for regularization
        )

        # Define new classification heads for action and severity, connected to the shared head
        self.action_head = nn.Linear(hidden_dim, 8)  # 8 classes for action
        self.severity_head = nn.Linear(hidden_dim, 4) # 4 classes for severity

    def forward(self, x):
        # Get the features from the backbone model
        features = self.model(x)

        # Pass features through the shared custom head
        processed_features = self.shared_head(features)

        # Pass the processed features through each of your new heads
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
    num_frames = 16 # Example number of frames per clip
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, 3, num_frames, height, width)
    print(f"Created dummy input with shape: {dummy_input.shape}")

    # Pass dummy input through the model
    try:
        action_logits, severity_logits = model(dummy_input)
        print("Model forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        exit()

    # Assert output shapes
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
