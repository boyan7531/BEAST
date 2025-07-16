import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MVFoulsModel
from dataset import MVFoulsDataset, custom_collate_fn
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score # Import metrics
import torch.nn.functional as F
import random
import numpy as np
import torch.cuda.amp as amp
from collections import Counter
import torch.optim.lr_scheduler # Import lr_scheduler
from transform import get_train_transforms, get_val_transforms # Import new transform functions
from torchvision.models.video import MViT_V2_S_Weights # Needed to get model input size

# Mixup function for data augmentation
def mixup_data(x_list, y_action, y_severity, alpha=0.2):
    """Apply mixup augmentation to video data and labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = len(x_list)
    index = torch.randperm(batch_size)

    # Mix the video data
    mixed_x = []
    for i in range(batch_size):
        # For each video tensor in the batch, mix with another
        mixed_video = lam * x_list[i] + (1 - lam) * x_list[index[i]]
        mixed_x.append(mixed_video)

    # Mix the labels
    y_action_a, y_action_b = y_action, y_action[index]
    y_severity_a, y_severity_b = y_severity, y_severity[index]
    
    return mixed_x, y_action_a, y_action_b, y_severity_a, y_severity_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Define Focal Loss with Label Smoothing
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean', alpha=None, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # New: per-class alpha tensor (e.g., higher for minorities)
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs are logits
        # targets are class indices
        num_classes = inputs.size(1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Create smoothed labels
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Use KL divergence for smoothed labels
            log_probs = F.log_softmax(inputs, dim=1)
            loss = F.kl_div(log_probs, smooth_targets, reduction='none').sum(dim=1)
            
            # Calculate pt for focal term
            pt = F.softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard focal loss implementation
            logpt = F.log_softmax(inputs, dim=1)
            loss = F.nll_loss(logpt, targets, reduction='none', weight=self.weight)
            pt = torch.exp(logpt.gather(1, targets.unsqueeze(1))).squeeze(1)
        
        # Apply the modulating factor
        focal_term = (1 - pt) ** self.gamma
        F_loss = focal_term * loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For all GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Hyperparameters
# EPOCHS = 10
# BATCH_SIZE = 4
# LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configuration
# DATA_FOLDER = "mvfouls"
# TRAIN_SPLIT = "train"
# VAL_SPLIT = "valid"
# START_FRAME = 67
# END_FRAME = 82


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Train MVFoulsModel")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # parser.add_argument('--data_folder', type=str, default='mvfouls', help='Path to the dataset folder')
    # parser.add_argument('--train_split', type=str, default='train', help='Name of the training split folder')
    # parser.add_argument('--val_split', type=str, default='valid', help='Name of the validation split folder')
    # parser.add_argument('--start_frame', type=int, default=67, help='Start frame for video clips')
    # parser.add_argument('--end_frame', type=int, default=82, help='End frame for video clips')
    parser.add_argument('--test_batches', type=int, default=0, help='Number of batches to run for testing (0 for full run)')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of CrossEntropyLoss')
    # parser.add_argument('--focal_loss_alpha', type=float, default=1.0, help='Alpha parameter for Focal Loss') # Removed this argument
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of batches to accumulate gradients over')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Use arguments for hyperparameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    TEST_BATCHES = args.test_batches
    USE_FOCAL_LOSS = args.use_focal_loss
    # FOCAL_LOSS_ALPHA = args.focal_loss_alpha # Removed
    FOCAL_LOSS_GAMMA = args.focal_loss_gamma
    NUM_WORKERS = args.num_workers
    ACCUMULATION_STEPS = args.accumulation_steps
    
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Using Focal Loss: {USE_FOCAL_LOSS}")
    print(f"Focal Loss Gamma: {FOCAL_LOSS_GAMMA}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Accumulation Steps: {ACCUMULATION_STEPS}")

    # Hardcoded values for removed arguments
    DATA_FOLDER = "mvfouls"
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "valid"
    START_FRAME = 67
    END_FRAME = 82

    # Determine the model input size dynamically from the MViT model weights
    # MViT_V2_S_Weights.KINETICS400_V1.transforms()._size is the expected input size
    # weights = MViT_V2_S_Weights.KINETICS400_V1 # Removed as _size attribute is not directly available
    MODEL_INPUT_SIZE = (224, 224) # MViT models typically use 224x224 input resolution

    # For Windows, num_workers must be 0 for DataLoader to avoid multiprocessing issues
    if os.name == 'nt':
        if NUM_WORKERS > 0:
            print("Warning: num_workers > 0 is not supported on Windows. Setting num_workers to 0.")
        NUM_WORKERS = 0

    print(f"Using device: {DEVICE}")

    # Initialize GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler(device='cuda') # Corrected for newer PyTorch versions

    # Helper function to calculate DISTRIBUTION-AWARE class weights
    def calculate_class_weights(labels_list, num_classes, epoch=0):
        # labels_list is a list of one-hot encoded tensors like [tensor([0, 1, 0]), tensor([1, 0, 0])] for a batch
        # First, concatenate all labels and convert one-hot to class indices
        all_labels_indices = torch.cat(labels_list, dim=0).argmax(dim=1).cpu().numpy()
        
        # Count occurrences of each class
        class_counts = Counter(all_labels_indices)
        total_samples = len(all_labels_indices)
        
        # Calculate class frequencies
        class_frequencies = np.array([class_counts.get(i, 0) / total_samples for i in range(num_classes)])
        
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            freq = class_frequencies[i]
            if freq > 0:
                # Distribution-aware weighting strategy
                if freq > 0.3:  # Majority classes (>30% of data)
                    # Conservative weights for majority classes: 0.9x to 1.1x
                    weights[i] = max(0.9, min(1.1, 1.0 / (freq ** 0.3)))
                elif freq > 0.05:  # Medium classes (5-30% of data)
                    # Moderate weights for medium classes: 1.0x to 1.8x
                    weights[i] = min(1.8, 1.0 / (freq ** 0.5))
                else:  # Minority classes (<5% of data)
                    # Controlled weights for minorities: 1.5x to 2.2x (not 3.0x)
                    weights[i] = min(2.2, max(1.5, 1.0 / (freq ** 0.6)))
            else:
                weights[i] = 1.0  # Default weight for missing classes
        
        return weights.to(DEVICE) # Move weights to the same device as the model

    # 2. Prepare Datasets and DataLoaders
    # Initialize training dataset and dataloader
    train_dataset = MVFoulsDataset(DATA_FOLDER, TRAIN_SPLIT, START_FRAME, END_FRAME, transform_model=get_train_transforms(MODEL_INPUT_SIZE))

    # Calculate class weights for action and severity labels in the training set
    num_action_classes = 8 # As defined in data_loader.py
    num_severity_classes = 4 # As defined in data_loader.py

    action_class_weights = calculate_class_weights(train_dataset.labels_action_list, num_action_classes)
    severity_class_weights = calculate_class_weights(train_dataset.labels_severity_list, num_severity_classes)
    
    print(f"Action class weights: {action_class_weights}")
    print(f"Severity class weights: {severity_class_weights}")

    # DISTRIBUTION-AWARE sampling strategy that respects majority classes
    all_action_labels = torch.cat(train_dataset.labels_action_list, dim=0).argmax(dim=1).cpu().numpy()
    all_severity_labels = torch.cat(train_dataset.labels_severity_list, dim=0).argmax(dim=1).cpu().numpy()
    
    # Calculate action class frequencies for sampling adjustment
    action_counts = Counter(all_action_labels)
    total_samples = len(all_action_labels)
    
    # Create CONSERVATIVE sampling weights based on action classes only
    sample_weights = []
    for j in range(len(train_dataset)):
        action_class = all_action_labels[j]
        freq = action_counts[action_class] / total_samples
        
        if freq > 0.3:  # Majority classes (Standing tackling: 44.76%)
            # Minimal adjustment for majority classes
            weight = max(0.8, min(1.2, 1.0 / (freq ** 0.2)))
        elif freq > 0.1:  # Medium classes (Tackling: 15.39%, Challenge: 14.49%, Holding: 11.77%)
            # Moderate adjustment for medium classes
            weight = min(1.5, 1.0 / (freq ** 0.4))
        else:  # Minority classes (<10%)
            # Controlled adjustment for minorities
            weight = min(1.8, max(1.2, 1.0 / (freq ** 0.5)))
        
        sample_weights.append(weight)

    # Create WeightedRandomSampler with conservative weights
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=custom_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Initialize validation dataset and dataloader
    val_dataset = MVFoulsDataset(DATA_FOLDER, VAL_SPLIT, START_FRAME, END_FRAME, transform_model=get_val_transforms(MODEL_INPUT_SIZE))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)


    # Check if datasets have samples
    if len(train_dataset) == 0:
        print("No samples in training dataset. Please ensure the dataset is correctly prepared and downloaded.")
        exit()
    
    print(f"Number of batches in training dataloader: {len(train_dataloader)}")
    print(f"Number of batches in validation dataloader: {len(val_dataloader)}")

    # 3. Initialize Model, Loss Functions, and Optimizer
    model = MVFoulsModel().to(DEVICE)
    
    if USE_FOCAL_LOSS:
        # Distribution-aware alpha values that complement the class weights
        # Lower alpha for majority classes, higher for minorities (but not extreme)
        action_alpha = torch.tensor([1.2, 0.9, 1.8, 1.4, 1.8, 2.0, 1.4, 2.2], device=DEVICE)  # Aligned with frequencies
        severity_alpha = torch.tensor([1.1, 0.9, 1.3, 2.0], device=DEVICE)
        
        # Use different label smoothing for majority vs minority classes
        criterion_action = FocalLoss(gamma=1.2, alpha=action_alpha, weight=action_class_weights, label_smoothing=0.05)
        criterion_severity = FocalLoss(gamma=1.2, alpha=severity_alpha, weight=severity_class_weights, label_smoothing=0.05)
        print(f"Using Focal Loss with gamma=1.2, distribution-aware alpha, class weights, and conservative label smoothing=0.05.")
    else:
        criterion_action = nn.CrossEntropyLoss(weight=action_class_weights) # Also pass weights to CrossEntropyLoss if not using Focal Loss
        criterion_severity = nn.CrossEntropyLoss(weight=severity_class_weights) # Also pass weights to CrossEntropyLoss if not using Focal Loss
        print("Using CrossEntropyLoss with per-class weights.")

    # Initialize optimizer with discriminative learning rates
    # Smaller learning rate for the pre-trained backbone
    # Standard learning rate for the newly added heads
    optimizer = optim.Adam([
        {'params': model.model.parameters(), 'lr': LEARNING_RATE * 0.1}, # 1/10th of the main LR for backbone
        {'params': model.shared_head.parameters()},
        {'params': model.attention_module.parameters()},
        {'params': model.action_head.parameters()},
        {'params': model.severity_head.parameters()}
    ], lr=LEARNING_RATE) # Default LR for custom heads

    # Initialize learning rate scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 4

    print("Training setup complete. Starting training loop...")

    # Create a directory for saving models if it doesn't exist
    MODEL_SAVE_DIR = "models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {MODEL_SAVE_DIR}/")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Add tqdm for progress bar
        tqdm_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        for i, (videos, action_labels, severity_labels, action_ids) in enumerate(tqdm_dataloader):
            # For testing, break after a specified number of batches if TEST_BATCHES > 0
            if TEST_BATCHES > 0 and i >= TEST_BATCHES:
                print(f"Reached {TEST_BATCHES} batches for testing, breaking training loop.")
                break

            # Move data to the appropriate device
            # videos is now a list of tensors, where each tensor corresponds to an action event
            videos = [v.to(DEVICE) for v in videos] # Apply .to(DEVICE) to each video in the list
            
            # Convert one-hot encoded labels to class indices
            # Labels are now (batch_size, num_classes) directly from custom_collate_fn
            action_labels = torch.argmax(action_labels, dim=1).to(DEVICE)
            severity_labels = torch.argmax(severity_labels, dim=1).to(DEVICE)

            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type='cuda'):
                action_logits, severity_logits = model(videos)

                # Calculate loss
                loss_action = criterion_action(action_logits, action_labels)
                loss_severity = criterion_severity(severity_logits, severity_labels)
                total_loss = loss_action + loss_severity # Combine losses

            # Normalize loss to account for accumulation
            total_loss = total_loss / ACCUMULATION_STEPS

            # Backward pass (gradients are accumulated)
            scaler.scale(total_loss).backward()

            running_loss += total_loss.item()
            tqdm_dataloader.set_postfix(loss=total_loss.item())

            # Perform optimizer step and zero gradients only after ACCUMULATION_STEPS batches
            if (i + 1) % ACCUMULATION_STEPS == 0 or (TEST_BATCHES > 0 and (i + 1) == TEST_BATCHES) or (i + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Clear gradients for the next accumulation cycle

        current_batches_processed = i + 1 if TEST_BATCHES == 0 else min(i + 1, TEST_BATCHES)
        avg_train_loss = running_loss / current_batches_processed if current_batches_processed > 0 else 0.0
        print(f"Epoch {epoch + 1} finished. Average Training Loss: {avg_train_loss:.4f}")

        # Save the model after each epoch
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # 5. Evaluation Loop
        print("\nStarting validation...")
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        all_action_labels = []
        all_predicted_actions = []
        all_severity_labels = []
        all_predicted_severities = []

        if len(val_dataset) == 0:
            print("No samples in validation dataset. Skipping validation.")
        else:
            with torch.no_grad(): # Disable gradient calculation for validation
                # Add tqdm for progress bar
                for i, (videos, action_labels, severity_labels, action_ids) in enumerate(tqdm(val_dataloader, desc="Validation")):
                    # For testing, break after a specified number of batches if TEST_BATCHES > 0
                    if TEST_BATCHES > 0 and i >= TEST_BATCHES:
                        print(f"Reached {TEST_BATCHES} batches for testing, breaking validation loop.")
                        break

                    videos = [v.to(DEVICE) for v in videos] # Apply .to(DEVICE) to each video in the list
                    action_labels_idx = torch.argmax(action_labels, dim=1).to(DEVICE)
                    severity_labels_idx = torch.argmax(severity_labels, dim=1).to(DEVICE)

                    with torch.amp.autocast(device_type='cuda'):
                        action_logits, severity_logits = model(videos)

                        loss_action = criterion_action(action_logits, action_labels_idx)
                        loss_severity = criterion_severity(severity_logits, severity_labels_idx)
                        total_loss = loss_action + loss_severity
                    val_running_loss += total_loss.item()

                    # Collect predictions and true labels for metrics
                    _, predicted_actions = torch.max(action_logits, 1)
                    _, predicted_severities = torch.max(severity_logits, 1)

                    all_action_labels.extend(action_labels_idx.cpu().numpy())
                    all_predicted_actions.extend(predicted_actions.cpu().numpy())
                    all_severity_labels.extend(severity_labels_idx.cpu().numpy()) # Corrected: Use true labels
                    all_predicted_severities.extend(predicted_severities.cpu().numpy())

            if len(all_action_labels) > 0:
                # Calculate and print metrics for action classification
                action_accuracy = accuracy_score(all_action_labels, all_predicted_actions)
                action_precision, action_recall, action_f1, _ = precision_recall_fscore_support(all_action_labels, all_predicted_actions, average='macro', zero_division=0)

                print(f"\nAction Classification Metrics:")
                print(f"  Accuracy: {action_accuracy:.4f}")
                print(f"  Macro Recall: {action_recall:.4f}")
                print(f"  Macro F1-score: {action_f1:.4f}")

                # Per-class action metrics to monitor overfitting
                action_precision_per_class, action_recall_per_class, action_f1_per_class, _ = precision_recall_fscore_support(all_action_labels, all_predicted_actions, average=None, zero_division=0)
                action_classes = ["Tackling", "Standing tackling", "High leg", "Holding", "Pushing", "Elbowing", "Challenge", "Dive"]
                print("  Per-class Action Recall:")
                for i, (cls, recall) in enumerate(zip(action_classes, action_recall_per_class)):
                    print(f"    {cls}: {recall:.3f}")

                # Calculate and print metrics for severity classification
                severity_accuracy = accuracy_score(all_severity_labels, all_predicted_severities)
                severity_precision, severity_recall, severity_f1, _ = precision_recall_fscore_support(all_severity_labels, all_predicted_severities, average='macro', zero_division=0)

                print(f"\nSeverity Classification Metrics:")
                print(f"  Accuracy: {severity_accuracy:.4f}")
                print(f"  Macro Recall: {severity_recall:.4f}")
                print(f"  Macro F1-score: {severity_f1:.4f}")
                
                combined_macro_recall = (action_recall + severity_recall) / 2
                combined_macro_f1 = (action_f1 + severity_f1) / 2
                print(f"Combined Macro Recall: {combined_macro_recall:.4f}")
                print(f"Combined Macro F1-score: {combined_macro_f1:.4f}")

                current_batches_processed_val = i + 1 if TEST_BATCHES == 0 else min(i + 1, TEST_BATCHES)
                avg_val_loss = val_running_loss / current_batches_processed_val if current_batches_processed_val > 0 else 0.0
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Step the learning rate scheduler with validation loss
                scheduler.step(avg_val_loss)
                
                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}")
                    
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                print("No samples processed in validation.")

    print("\nTraining complete!") # This final print is outside the loop
