import torchvision.transforms.v2 as transforms
import torch
from torchvision.models.video import MViT_V2_S_Weights

class PermuteTCHW(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2) # From (T, H, W, C) to (T, C, H, W)

class PermuteCTHW(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2, 3) # From (T, C, H, W) to (C, T, H, W)

def get_train_transforms(model_input_size, use_augmentation=True):
    # Manual implementation of official MViT preprocessing with original paper's augmentations
    if use_augmentation:
        return transforms.Compose([
            # First, permute to (T, C, H, W) for torchvision transforms
            PermuteTCHW(),
            
            # Original paper's aggressive augmentation strategy
            transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(),
            
            # Official MViT preprocessing pipeline
            transforms.Resize([256], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop([224, 224]),  # Random crop for training
            
            # Official MViT normalization
            transforms.ConvertImageDtype(torch.float32), # Rescale to [0.0, 1.0]
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            
            # Final permute to (C, T, H, W) as expected by MViT
            PermuteCTHW(),
        ])
    else:
        return transforms.Compose([
            # First, permute to (T, C, H, W) for torchvision transforms
            PermuteTCHW(),
            
            # Official MViT preprocessing pipeline (no augmentation)
            transforms.Resize([256], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop([224, 224]),
            
            # Official MViT normalization
            transforms.ConvertImageDtype(torch.float32), # Rescale to [0.0, 1.0]
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            
            # Final permute to (C, T, H, W) as expected by MViT
            PermuteCTHW(),
        ])

def get_val_transforms(model_input_size):
    # Official MViT preprocessing pipeline (validation/inference)
    return transforms.Compose([
        # First, permute to (T, C, H, W) for torchvision transforms
        PermuteTCHW(),
        
        # Official MViT preprocessing pipeline
        transforms.Resize([256], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop([224, 224]),
        
        # Official MViT normalization
        transforms.ConvertImageDtype(torch.float32), # Rescale to [0.0, 1.0]
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        
        # Final permute to (C, T, H, W) as expected by MViT
        PermuteCTHW(),
    ])
