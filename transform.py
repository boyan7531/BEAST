import torchvision.transforms.v2 as transforms
import torch
from torchvision.models.video import MViT_V2_S_Weights

def get_train_transforms(model_input_size):
    # Get mean and std from MViT weights for normalization
    weights = MViT_V2_S_Weights.KINETICS400_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        # These transforms operate on (T, H, W, C) tensor (or PIL image if input)
        transforms.RandomResizedCrop(model_input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        
        # Convert to float and permute to (C, T, H, W) as expected by models like MViT
        transforms.ConvertImageDtype(torch.float32), # Converts to float [0, 1]
        transforms.Normalize(mean=mean, std=std),
        transforms.Permute([3, 0, 1, 2]), # Change from (T, H, W, C) to (C, T, H, W)
    ])

def get_val_transforms(model_input_size):
    # Get mean and std from MViT weights for normalization
    weights = MViT_V2_S_Weights.KINETICS400_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        # These transforms operate on (T, H, W, C) tensor (or PIL image if input)
        transforms.Resize(model_input_size), # Resize to model input size
        
        # Convert to float and permute to (C, T, H, W) as expected by models like MViT
        transforms.ConvertImageDtype(torch.float32), # Converts to float [0, 1]
        transforms.Normalize(mean=mean, std=std),
        transforms.Permute([3, 0, 1, 2]), # Change from (T, H, W, C) to (C, T, H, W)
    ])
