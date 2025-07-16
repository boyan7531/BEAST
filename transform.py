import torchvision.transforms.v2 as transforms
import torch
from torchvision.models.video import MViT_V2_S_Weights

def get_train_transforms(model_input_size):
    # Get mean and std from MViT weights for normalization
    weights = MViT_V2_S_Weights.KINETICS400_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        # First, permute to (T, C, H, W) for torchvision transforms
        lambda x: x.permute(0, 3, 1, 2), # From (T, H, W, C) to (T, C, H, W)
        
        # These transforms now operate on (T, C, H, W) tensor (or PIL image if input)
        transforms.RandomResizedCrop(model_input_size, scale=(0.9, 1.0)),  # Less aggressive cropping
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced flip probability
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),  # Gentler color changes
        
        # Convert to float and normalize
        transforms.ConvertImageDtype(torch.float32), # Converts to float [0, 1]
        transforms.Normalize(mean=mean, std=std),
        
        # Final permute to (C, T, H, W) as expected by models like MViT
        lambda x: x.permute(1, 0, 2, 3), # From (T, C, H, W) to (C, T, H, W)
    ])

def get_val_transforms(model_input_size):
    # Get mean and std from MViT weights for normalization
    weights = MViT_V2_S_Weights.KINETICS400_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        # First, permute to (T, C, H, W) for torchvision transforms
        lambda x: x.permute(0, 3, 1, 2), # From (T, H, W, C) to (T, C, H, W)
        
        # These transforms now operate on (T, C, H, W) tensor (or PIL image if input)
        transforms.Resize(model_input_size), # Resize to model input size
        
        # Convert to float and normalize
        transforms.ConvertImageDtype(torch.float32), # Converts to float [0, 1]
        transforms.Normalize(mean=mean, std=std),
        
        # Final permute to (C, T, H, W) as expected by models like MViT
        lambda x: x.permute(1, 0, 2, 3), # From (T, C, H, W) to (C, T, H, W)
    ])
