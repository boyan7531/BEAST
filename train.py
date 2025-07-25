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

# Smart Rebalancing System imports
from smart_rebalancer import SmartRebalancer, RebalancingConfig, CurriculumConfig, CurriculumStage
from rebalancing_integration import (
    extract_performance_metrics,
    create_rebalancer_from_dataset,
    log_rebalancing_status,
    update_focal_loss_from_rebalancer,
    get_mixup_params_from_rebalancer,
    create_curriculum_dataloader,
    update_curriculum_loss_functions,
    ACTION_CLASS_NAMES,
    SEVERITY_CLASS_NAMES
)

# Mixup function for data augmentation
def mixup_data(x_list, y_action, y_severity, alpha=0.2):
    """Apply mixup augmentation to video data and labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = len(x_list)
    index = torch.randperm(batch_size)

    # Mix the video data - handle different clip lengths
    mixed_x = []
    for i in range(batch_size):
        video_a = x_list[i]
        video_b = x_list[index[i]]
        
        # Handle different numbers of clips by taking minimum
        min_clips = min(video_a.shape[0], video_b.shape[0])
        
        # Truncate both videos to same length and mix
        video_a_truncated = video_a[:min_clips]
        video_b_truncated = video_b[:min_clips]
        
        mixed_video = lam * video_a_truncated + (1 - lam) * video_b_truncated
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

# Asymmetric Focal Loss for extreme class imbalance
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, alpha=0.25, reduction='mean', weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        # Convert to probabilities
        p = torch.sigmoid(inputs)
        
        # For multi-class, we need to handle this differently
        if inputs.dim() > 1 and inputs.size(1) > 1:
            # Multi-class case - convert to one-hot and use cross-entropy style
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
            p_t = torch.exp(-ce_loss)
            
            # Apply asymmetric focusing
            alpha_t = self.alpha
            focal_weight = alpha_t * (1 - p_t) ** self.gamma_pos
            
            focal_loss = focal_weight * ce_loss
        else:
            # Binary case
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
            # Asymmetric focusing
            focal_weight = alpha_t * ((1 - p_t) ** self.gamma_pos) * targets + \
                          (1 - alpha_t) * (p_t ** self.gamma_neg) * (1 - targets)
            
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Stratified Batch Sampler to ensure minority classes in every batch
class StratifiedBatchSampler:
    def __init__(self, labels, batch_size, min_minority_per_batch=1):
        self.labels = labels
        self.batch_size = batch_size
        self.min_minority_per_batch = min_minority_per_batch
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            class_id = label.item() if torch.is_tensor(label) else label
            if class_id not in self.class_indices:
                self.class_indices[class_id] = []
            self.class_indices[class_id].append(idx)
        
        # Identify minority classes (< 5% of data)
        total_samples = len(labels)
        self.minority_classes = []
        for class_id, indices in self.class_indices.items():
            if len(indices) / total_samples < 0.05:
                self.minority_classes.append(class_id)
        
        print(f"Minority classes identified: {self.minority_classes}")
        
    def __iter__(self):
        # Create batches ensuring minority representation
        all_indices = list(range(len(self.labels)))
        random.shuffle(all_indices)
        
        batches = []
        i = 0
        while i < len(all_indices):
            batch = []
            
            # First, add minority samples if available
            minority_added = 0
            for class_id in self.minority_classes:
                if minority_added < self.min_minority_per_batch and self.class_indices[class_id]:
                    minority_idx = random.choice(self.class_indices[class_id])
                    if minority_idx in all_indices[i:]:
                        batch.append(minority_idx)
                        all_indices.remove(minority_idx)
                        minority_added += 1
            
            # Fill the rest of the batch
            remaining_slots = self.batch_size - len(batch)
            end_idx = min(i + remaining_slots, len(all_indices))
            batch.extend(all_indices[i:end_idx])
            i = end_idx
            
            if len(batch) > 0:
                batches.append(batch)
        
        return iter(batches)
    
    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For all GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rebalancer=None):
    """Load model checkpoint and resume training"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        # New checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load rebalancer state if available
        if rebalancer is not None and 'rebalancer_state' in checkpoint:
            rebalancer.load_state_dict(checkpoint['rebalancer_state'])
            print("Loaded rebalancer state from checkpoint")
    else:
        # Legacy checkpoint (just model weights)
        model.load_state_dict(checkpoint)
        start_epoch = 0
        best_val_loss = float('inf')
    
    print(f"Loaded checkpoint from epoch {start_epoch-1}, best val loss: {best_val_loss}")
    return start_epoch, best_val_loss

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
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of batches to accumulate gradients over')
    parser.add_argument('--exclude_classes', type=str, default='', 
                        help='Comma-separated list of classes to exclude from training (e.g., "action_4,action_7,severity_3" for Pushing,Dive,RedCard)')
    parser.add_argument('--exclude_problematic', action='store_true',
                        help='Exclude the most problematic classes: Pushing, Dive, and Red Card from training')
    
    # Aggregation method argument
    parser.add_argument('--aggregation', type=str, default='attention', choices=['max', 'mean', 'attention'], 
                        help='Aggregation method for multi-view clips: max, mean, or attention')
    
    # Smart Rebalancing System arguments
    parser.add_argument('--use_smart_rebalancing', action='store_true', help='Enable smart rebalancing system')
    parser.add_argument('--min_class_recall', type=float, default=0.3, help='Minimum acceptable recall per class')
    parser.add_argument('--target_macro_recall', type=float, default=0.7, help='Target macro-averaged recall')
    parser.add_argument('--adaptation_rate', type=float, default=0.1, help='Rate of rebalancing adaptations')
    
    # Curriculum Learning arguments
    parser.add_argument('--use_curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--curriculum_easy_epochs', type=int, default=8, help='Epochs for easy-only stage')
    parser.add_argument('--curriculum_medium_epochs', type=int, default=6, help='Epochs for medium introduction')
    parser.add_argument('--curriculum_full_epochs', type=int, default=6, help='Epochs for full curriculum')
    
    # Checkpoint and fine-tuning arguments
    parser.add_argument('--load_checkpoint', type=str, help='Load model checkpoint to continue training')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone during training')

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
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Accumulation Steps: {ACCUMULATION_STEPS}")
    
    # Smart Rebalancing System configuration
    USE_SMART_REBALANCING = args.use_smart_rebalancing
    print(f"Smart Rebalancing: {'Enabled' if USE_SMART_REBALANCING else 'Disabled'}")
    if USE_SMART_REBALANCING:
        print(f"  Min Class Recall: {args.min_class_recall}")
        print(f"  Target Macro Recall: {args.target_macro_recall}")
        print(f"  Adaptation Rate: {args.adaptation_rate}")
    
    # Curriculum Learning configuration
    USE_CURRICULUM = args.use_curriculum
    print(f"Curriculum Learning: {'Enabled' if USE_CURRICULUM else 'Disabled'}")
    if USE_CURRICULUM:
        print(f"  Easy Stage Epochs: {args.curriculum_easy_epochs}")
        print(f"  Medium Stage Epochs: {args.curriculum_medium_epochs}")
        print(f"  Full Stage Epochs: {args.curriculum_full_epochs}")
    
    # Checkpoint loading
    LOAD_CHECKPOINT = args.load_checkpoint
    FREEZE_BACKBONE = args.freeze_backbone
    if LOAD_CHECKPOINT:
        print(f"Will load checkpoint from: {LOAD_CHECKPOINT}")
    if FREEZE_BACKBONE:
        print("Backbone will be frozen during training")

    # Hardcoded values for removed arguments
    DATA_FOLDER = "mvfouls"
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "test"
    START_FRAME = 63
    END_FRAME = 86

    # Determine the model input size dynamically from the MViT model weights
    # MViT_V2_S_Weights.KINETICS400_V1.transforms()._size is the expected input size
    # weights = MViT_V2_S_Weights.KINETICS400_V1 # Removed as _size attribute is not directly available
    MODEL_INPUT_SIZE = (224, 224) # Official MViT input size (after resize→crop pipeline)

    # For Windows, num_workers must be 0 for DataLoader to avoid multiprocessing issues
    if os.name == 'nt':
        if NUM_WORKERS > 0:
            print("Warning: num_workers > 0 is not supported on Windows. Setting num_workers to 0.")
        NUM_WORKERS = 0

    print(f"Using device: {DEVICE}")

    # Parse class exclusion settings
    excluded_action_classes = set()
    excluded_severity_classes = set()
    
    if args.exclude_problematic:
        # Exclude the most problematic classes: Pushing (4), Dive (7), Red Card (3)
        excluded_action_classes.update([4, 7])  # Pushing, Dive
        excluded_severity_classes.update([3])   # Red Card
        print("Excluding problematic classes from training: Pushing, Dive, Red Card")
    
    if args.exclude_classes:
        # Parse custom exclusion list (e.g., "action_4,action_7,severity_3")
        for class_spec in args.exclude_classes.split(','):
            class_spec = class_spec.strip()
            if class_spec.startswith('action_'):
                class_id = int(class_spec.split('_')[1])
                excluded_action_classes.add(class_id)
            elif class_spec.startswith('severity_'):
                class_id = int(class_spec.split('_')[1])
                excluded_severity_classes.add(class_id)
        print(f"Custom excluded action classes: {sorted(excluded_action_classes)}")
        print(f"Custom excluded severity classes: {sorted(excluded_severity_classes)}")
    
    if excluded_action_classes or excluded_severity_classes:
        print(f"Final excluded action classes: {sorted(excluded_action_classes)}")
        print(f"Final excluded severity classes: {sorted(excluded_severity_classes)}")
        print("Note: Model will be trained without these classes but evaluated on ALL original classes")

    # Initialize GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler(device='cuda') # Corrected for newer PyTorch versions

    # Helper function to calculate AGGRESSIVE SEVERITY-AWARE class weights
    def calculate_class_weights(labels_list, num_classes, task_type="action"):
        # labels_list is a list of one-hot encoded tensors like [tensor([0, 1, 0]), tensor([1, 0, 0])] for a batch
        # First, concatenate all labels and convert one-hot to class indices
        all_labels_indices = torch.cat(labels_list, dim=0).argmax(dim=1).cpu().numpy()
        
        # Count occurrences of each class
        class_counts = Counter(all_labels_indices)
        total_samples = len(all_labels_indices)
        
        # Calculate class frequencies and print for debugging
        class_frequencies = np.array([class_counts.get(i, 0) / total_samples for i in range(num_classes)])
        print(f"{task_type.capitalize()} class frequencies: {class_frequencies}")
        
        weights = torch.zeros(num_classes)
        
        if task_type == "severity":
            # CONSERVATIVE rebalancing for severity classification (same as action)
            for i in range(num_classes):
                freq = class_frequencies[i]
                if freq > 0:
                    if freq > 0.4:  # Very dominant classes (>40%) - Offence + No Card
                        weights[i] = max(0.9, 1.0 / (freq ** 0.2))
                    elif freq > 0.25:  # Major classes (25-40%) - Yellow Card
                        weights[i] = max(1.0, min(1.8, 1.0 / (freq ** 0.4)))
                    elif freq > 0.1:  # Medium classes (10-25%) - No Offence
                        weights[i] = min(2.0, 1.0 / (freq ** 0.5))
                    else:  # Very small classes (<10%) - Red Card
                        weights[i] = min(2.5, max(1.8, 1.0 / (freq ** 0.7)))
                else:
                    weights[i] = 1.0
        else:
            # Conservative rebalancing for action classification
            for i in range(num_classes):
                freq = class_frequencies[i]
                if freq > 0:
                    if freq > 0.4:  # Very dominant classes (>40%)
                        weights[i] = max(0.9, 1.0 / (freq ** 0.2))
                    elif freq > 0.25:  # Major classes (25-40%)
                        weights[i] = max(1.0, min(1.3, 1.0 / (freq ** 0.4)))
                    elif freq > 0.1:  # Medium classes (10-25%)
                        weights[i] = min(1.6, 1.0 / (freq ** 0.5))
                    elif freq > 0.03:  # Small classes (3-10%)
                        weights[i] = min(2.0, 1.0 / (freq ** 0.6))
                    else:  # Very small classes (<3%)
                        weights[i] = min(2.5, max(1.8, 1.0 / (freq ** 0.7)))
                else:
                    weights[i] = 1.0
        
        print(f"Calculated {task_type} weights: {weights}")
        return weights.to(DEVICE)

    # 2. Prepare Datasets and DataLoaders
    # Initialize training dataset and dataloader
    train_dataset = MVFoulsDataset(DATA_FOLDER, TRAIN_SPLIT, START_FRAME, END_FRAME, transform_model=get_train_transforms(MODEL_INPUT_SIZE))
    
    # Filter training dataset if class exclusion is enabled
    if excluded_action_classes or excluded_severity_classes:
        print(f"Filtering training dataset to exclude classes...")
        original_train_size = len(train_dataset)
        
        # Fast filtering using pre-loaded labels (no video loading needed)
        filtered_indices = []
        for i in range(len(train_dataset.labels_action_list)):
            action_label = train_dataset.labels_action_list[i]
            severity_label = train_dataset.labels_severity_list[i]
            
            # Convert one-hot to class indices
            action_class = torch.argmax(action_label).item()
            severity_class = torch.argmax(severity_label).item()
            
            # Keep sample if it doesn't contain excluded classes
            keep_sample = True
            if action_class in excluded_action_classes:
                keep_sample = False
            if severity_class in excluded_severity_classes:
                keep_sample = False
                
            if keep_sample:
                filtered_indices.append(i)
        
        # Create filtered dataset (fast - no video loading)
        train_dataset.data_list = [train_dataset.data_list[i] for i in filtered_indices]
        train_dataset.labels_action_list = [train_dataset.labels_action_list[i] for i in filtered_indices]
        train_dataset.labels_severity_list = [train_dataset.labels_severity_list[i] for i in filtered_indices]
        train_dataset.length = len(train_dataset.data_list)
        
        print(f"Training dataset filtered: {original_train_size} → {len(train_dataset)} samples")
        print(f"Removed {original_train_size - len(train_dataset)} samples with excluded classes")

    # Calculate class weights for action and severity labels in the training set
    num_action_classes = 8 # As defined in data_loader.py
    num_severity_classes = 4 # As defined in data_loader.py

    action_class_weights = calculate_class_weights(train_dataset.labels_action_list, num_action_classes, "action")
    severity_class_weights = calculate_class_weights(train_dataset.labels_severity_list, num_severity_classes, "severity")
    
    print(f"Action class weights: {action_class_weights}")
    print(f"Severity class weights: {severity_class_weights}")

    # ULTRA-AGGRESSIVE SEVERITY-FOCUSED sampling strategy
    all_action_labels = torch.cat(train_dataset.labels_action_list, dim=0).argmax(dim=1).cpu().numpy()
    all_severity_labels = torch.cat(train_dataset.labels_severity_list, dim=0).argmax(dim=1).cpu().numpy()
    
    # Calculate severity class frequencies for ultra-aggressive sampling adjustment
    severity_counts = Counter(all_severity_labels)
    total_samples = len(all_severity_labels)
    
    print(f"Severity class distribution: {dict(severity_counts)}")
    print(f"Severity frequencies: {[(i, severity_counts[i]/total_samples) for i in range(4)]}")
    
    # Create HYBRID sampling weights - combine severity and action class info
    sample_weights = []
    action_counts = Counter(all_action_labels)
    
    for j in range(len(train_dataset)):
        severity_class = all_severity_labels[j]
        action_class = all_action_labels[j]
        
        # Base weight from severity
        severity_freq = severity_counts[severity_class] / total_samples
        if severity_class == 0:  # No Offence (13.11%)
            severity_weight = min(2.0, max(1.3, 1.0 / (severity_freq ** 0.5)))
        elif severity_class == 1:  # Offence + No Card (56.19%)
            severity_weight = 1.0
        elif severity_class == 2:  # Yellow Card (29.54%)
            severity_weight = min(1.8, max(1.2, 1.0 / (severity_freq ** 0.4)))
        else:  # Red Card (1.16%)
            severity_weight = min(4.0, max(2.0, 1.0 / (severity_freq ** 0.6)))
        
        # ULTRA-AGGRESSIVE boost for zero-recall action classes (45%+ target)
        action_freq = action_counts[action_class] / total_samples
        if action_class in [4, 7]:  # Pushing, Dive (persistent 0% recall)
            action_boost = 2.5  # Massive boost for zero-recall classes
        elif action_freq < 0.05:  # Other rare action classes
            action_boost = 1.8  # Strong boost for rare classes
        elif action_freq < 0.15:  # Medium rare classes
            action_boost = 1.3  # Moderate boost
        else:
            action_boost = 1.0
        
        weight = severity_weight * action_boost
        
        sample_weights.append(weight)

    # Create WeightedRandomSampler with aggressive severity-based weights
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"Severity-based sampling weights range: {min(sample_weights):.2f} to {max(sample_weights):.2f}")
    print(f"Red Card samples will be sampled ~{max(sample_weights)/min(sample_weights):.1f}x more frequently")

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

    # Initialize Smart Rebalancing System if enabled
    rebalancer = None
    if USE_SMART_REBALANCING:
        print("\nInitializing Smart Rebalancing System...")
        config = RebalancingConfig(
            min_class_recall=args.min_class_recall,
            target_macro_recall=args.target_macro_recall,
            adaptation_rate=args.adaptation_rate
        )
        
        # Initialize curriculum config if enabled
        curriculum_config = None
        if USE_CURRICULUM:
            curriculum_config = CurriculumConfig(
                easy_stage_epochs=args.curriculum_easy_epochs,
                medium_stage_epochs=args.curriculum_medium_epochs,
                full_stage_epochs=args.curriculum_full_epochs
            )
        
        rebalancer = create_rebalancer_from_dataset(
            train_dataset,
            config=config,
            save_dir="rebalancing_logs",
            excluded_action_classes=excluded_action_classes,
            excluded_severity_classes=excluded_severity_classes,
            curriculum_config=curriculum_config
        )
        print("Smart Rebalancing System initialized successfully!")

    # 3. Initialize Model, Loss Functions, and Optimizer
    model = MVFoulsModel(aggregation=args.aggregation).to(DEVICE)
    print(f"Using {args.aggregation} aggregation method")
    
    if USE_FOCAL_LOSS:
        # AGGRESSIVE alpha values for +5% macro recall improvement
        action_alpha = torch.tensor([1.0, 0.8, 2.5, 2.0, 2.5, 2.0, 2.0, 2.5], device=DEVICE)  # Boost High leg, Holding, Challenge heavily
        severity_alpha = torch.tensor([2.2, 0.85, 1.2, 3.0], device=DEVICE)  # Boost No Offence significantly, reduce Yellow Card dominance
        
        # Initialize with ultra-aggressive values (will be updated dynamically)
        criterion_action = FocalLoss(gamma=1.2, alpha=action_alpha, weight=action_class_weights, label_smoothing=0.05)
        criterion_severity = FocalLoss(gamma=2.8, alpha=severity_alpha, weight=severity_class_weights, label_smoothing=0.08)
        print(f"Using CONSERVATIVE Focal Loss for stable 45%+: Action[Pushing=1.8, Dive=2.0], Severity[No Offence=1.6, Yellow=1.5, Red=2.2]")
    else:
        criterion_action = nn.CrossEntropyLoss(weight=action_class_weights) # Also pass weights to CrossEntropyLoss if not using Focal Loss
        criterion_severity = nn.CrossEntropyLoss(weight=severity_class_weights) # Also pass weights to CrossEntropyLoss if not using Focal Loss
        print("Using CrossEntropyLoss with per-class weights.")

    
    # Initialize optimizer with TARGETED learning rates for severity-focused training
    # Separate parameters for severity head to give it more learning capacity
    severity_params = list(model.severity_head.parameters())
    severity_param_ids = {id(p) for p in severity_params}
    other_params = [p for p in model.parameters() if id(p) not in severity_param_ids]
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': LEARNING_RATE},
        {'params': severity_params, 'lr': LEARNING_RATE * 1.5}  # 50% higher LR for severity head
    ])
    
    print(f"Using targeted learning rates: Base={LEARNING_RATE}, Severity Head={LEARNING_RATE * 1.5}")

    # Initialize learning rate scheduler - conservative plateau-based reduction
    if USE_CURRICULUM:
        # ReduceLROnPlateau monitors combined macro recall - only reduces when performance plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=4, min_lr=1e-5, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler: mode=max (combined macro recall), factor=0.7, patience=4, min_lr=1e-5")
    else:
        # Less aggressive StepLR for non-curriculum training
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        print(f"Using StepLR scheduler: step_size=10, gamma=0.5 (single LR reduction at epoch 10)")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    start_epoch = 0

    # Load checkpoint if specified
    if LOAD_CHECKPOINT:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, LOAD_CHECKPOINT, rebalancer
        )

    print("Training setup complete. Starting training loop...")

    # Create a directory for saving models if it doesn't exist
    MODEL_SAVE_DIR = "models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {MODEL_SAVE_DIR}/")

    # 4. Training Loop
    previous_stage = None
    
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # Handle curriculum learning - recreate dataloader every epoch for proper shuffling
        current_stage = None
        if USE_CURRICULUM and USE_SMART_REBALANCING and rebalancer is not None:
            current_stage = rebalancer.get_current_curriculum_stage(epoch)
            
            # Always recreate dataloader for curriculum learning to ensure proper shuffling
            stage_changed = current_stage != previous_stage
            if stage_changed:
                print(f"\n=== Curriculum Stage Change: {current_stage.value} ===")
            else:
                print(f"\n=== Curriculum Epoch {epoch+1} (Stage: {current_stage.value}) ===")
            
            # Create new dataloader with curriculum filtering (every epoch for diversity)
            current_dataloader = create_curriculum_dataloader(
                train_dataset, rebalancer, epoch, BATCH_SIZE,
                collate_fn=custom_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
            )
            
            # Update loss functions with curriculum weights
            if USE_FOCAL_LOSS:
                criterion_action, criterion_severity = update_curriculum_loss_functions(
                    rebalancer, epoch, DEVICE, FocalLoss
                )
            else:
                action_weights = rebalancer.get_curriculum_class_weights(epoch, 'action', DEVICE)
                severity_weights = rebalancer.get_curriculum_class_weights(epoch, 'severity', DEVICE)
                criterion_action = nn.CrossEntropyLoss(weight=action_weights)
                criterion_severity = nn.CrossEntropyLoss(weight=severity_weights)
            
            if stage_changed:
                print(f"Updated dataloader and loss functions for new stage: {current_stage.value}")
            else:
                print(f"Recreated dataloader for epoch diversity (stage: {current_stage.value})")
            previous_stage = current_stage
        else:
            # Use original dataloader if curriculum learning is disabled
            current_dataloader = train_dataloader
        
        # Freeze backbone if in fine-tuning stage and flag is set
        if (FREEZE_BACKBONE and USE_CURRICULUM and USE_SMART_REBALANCING and 
            rebalancer is not None and current_stage == CurriculumStage.FINE_TUNING):
            print("Freezing backbone for fine-tuning stage")
            for param in model.model.parameters():  # MViT backbone
                param.requires_grad = False
            for param in model.shared_feature_head.parameters():
                param.requires_grad = True
            for param in model.action_head.parameters():
                param.requires_grad = True
            for param in model.severity_head.parameters():
                param.requires_grad = True
        
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Get mixup parameters from smart rebalancer if enabled
        use_mixup = False
        mixup_alpha = 0.2
        if USE_SMART_REBALANCING and rebalancer is not None:
            use_mixup, mixup_alpha = get_mixup_params_from_rebalancer(rebalancer, epoch)
        
        # Add tqdm for progress bar
        tqdm_dataloader = tqdm(current_dataloader, desc=f"Epoch {epoch+1} Training")
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
            
            # Note: Class exclusion filtering is now done at dataset level, so no need to filter here

            # Apply mixup if recommended by smart rebalancer
            if use_mixup and mixup_alpha > 0:
                mixed_videos, action_labels_a, action_labels_b, severity_labels_a, severity_labels_b, lam = mixup_data(
                    videos, action_labels, severity_labels, mixup_alpha
                )
                
                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    action_logits, severity_logits = model(mixed_videos)

                    # Calculate mixup loss
                    loss_action = mixup_criterion(criterion_action, action_logits, action_labels_a, action_labels_b, lam)
                    loss_severity = mixup_criterion(criterion_severity, severity_logits, severity_labels_a, severity_labels_b, lam)
                    total_loss = loss_action + loss_severity # Combine losses
            else:
                # Standard forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    action_logits, severity_logits = model(videos)

                    # Calculate loss with DYNAMIC weighting based on performance
                    loss_action = criterion_action(action_logits, action_labels)
                    loss_severity = criterion_severity(severity_logits, severity_labels)
                    
                    # ULTRA-AGGRESSIVE dynamic loss weighting for 45%+ target with safety checks
                    if epoch < 3:
                        # Very early epochs: extreme severity focus
                        severity_weight = 3.0
                    elif epoch < 8:
                        # Early-mid epochs: strong severity focus  
                        severity_weight = 2.5
                    elif epoch < 15:
                        # Mid epochs: balanced but severity-leaning
                        severity_weight = 1.8
                    else:
                        # Later epochs: slight severity preference
                        severity_weight = 1.4
                    
                    # Safety mechanism: if validation loss explodes, reduce severity weight
                    if 'prev_val_loss' in locals() and val_loss > prev_val_loss * 1.5:
                        severity_weight = min(severity_weight, 1.2)
                        print(f"Safety: Reduced severity weight to {severity_weight} due to loss spike")
                    
                    total_loss = loss_action + (severity_weight * loss_severity)

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
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'curriculum_stage': current_stage.value if current_stage else None,
            'rebalancer_state': rebalancer.get_state_dict() if rebalancer else None,
            'training_args': {
                'use_curriculum': USE_CURRICULUM,
                'use_smart_rebalancing': USE_SMART_REBALANCING,
                'freeze_backbone': FREEZE_BACKBONE
            }
        }
        
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

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

                # Per-class severity metrics to monitor overfitting
                severity_precision_per_class, severity_recall_per_class, severity_f1_per_class, _ = precision_recall_fscore_support(all_severity_labels, all_predicted_severities, average=None, zero_division=0)
                severity_classes = ["No Offence", "Offence + No Card", "Offence + Yellow Card", "Offence + Red Card"]
                print("  Per-class Severity Recall:")
                for i, (cls, recall) in enumerate(zip(severity_classes, severity_recall_per_class)):
                    print(f"    {cls}: {recall:.3f}")
                
                combined_macro_recall = (action_recall + severity_recall) / 2
                combined_macro_f1 = (action_f1 + severity_f1) / 2
                print(f"Combined Macro Recall: {combined_macro_recall:.4f}")
                print(f"Combined Macro F1-score: {combined_macro_f1:.4f}")

                current_batches_processed_val = i + 1 if TEST_BATCHES == 0 else min(i + 1, TEST_BATCHES)
                avg_val_loss = val_running_loss / current_batches_processed_val if current_batches_processed_val > 0 else 0.0
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Smart Rebalancing System integration
                if USE_SMART_REBALANCING and rebalancer is not None:
                    # Extract performance metrics for rebalancer
                    action_metrics = extract_performance_metrics(
                        all_action_labels, all_predicted_actions, 
                        avg_val_loss, 'action', ACTION_CLASS_NAMES
                    )
                    severity_metrics = extract_performance_metrics(
                        all_severity_labels, all_predicted_severities, 
                        avg_val_loss, 'severity', SEVERITY_CLASS_NAMES
                    )
                    
                    # Update rebalancer with current performance
                    rebalancer.update_performance(epoch + 1, action_metrics, severity_metrics)
                    
                    # Log rebalancing status
                    log_rebalancing_status(rebalancer, epoch + 1)
                    
                    # Update loss functions based on rebalancer recommendations if using focal loss
                    if USE_FOCAL_LOSS:
                        criterion_action = update_focal_loss_from_rebalancer(
                            rebalancer, 'action', DEVICE, FocalLoss
                        )
                        criterion_severity = update_focal_loss_from_rebalancer(
                            rebalancer, 'severity', DEVICE, FocalLoss
                        )
                        print("Updated loss functions based on rebalancer recommendations")
                print(f"  Macro F1-score: {severity_f1:.4f}")

                # Per-class severity metrics to monitor overfitting
                severity_precision_per_class, severity_recall_per_class, severity_f1_per_class, _ = precision_recall_fscore_support(all_severity_labels, all_predicted_severities, average=None, zero_division=0)
                severity_classes = ["No Offence", "Offence + No Card", "Offence + Yellow Card", "Offence + Red Card"]
                print("  Per-class Severity Recall:")
                for i, (cls, recall) in enumerate(zip(severity_classes, severity_recall_per_class)):
                    print(f"    {cls}: {recall:.3f}")
                
                combined_macro_recall = (action_recall + severity_recall) / 2
                combined_macro_f1 = (action_f1 + severity_f1) / 2
                print(f"Combined Macro Recall: {combined_macro_recall:.4f}")
                print(f"Combined Macro F1-score: {combined_macro_f1:.4f}")

                current_batches_processed_val = i + 1 if TEST_BATCHES == 0 else min(i + 1, TEST_BATCHES)
                avg_val_loss = val_running_loss / current_batches_processed_val if current_batches_processed_val > 0 else 0.0
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Step the learning rate scheduler with appropriate metric
                if USE_CURRICULUM:
                    # ReduceLROnPlateau monitors combined macro recall (mode='max')
                    scheduler.step(combined_macro_recall)
                else:
                    # StepLR doesn't need a metric
                    scheduler.step()
                
                # Early stopping logic - use combined macro recall for curriculum learning
                if USE_CURRICULUM:
                    # For curriculum learning, track best combined macro recall
                    if 'best_combined_recall' not in locals():
                        best_combined_recall = 0.0
                    
                    if combined_macro_recall > best_combined_recall:
                        best_combined_recall = combined_macro_recall
                        patience_counter = 0
                        # Save best model checkpoint
                        best_checkpoint = checkpoint.copy()
                        best_checkpoint['best_combined_recall'] = best_combined_recall
                        
                        best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_curriculum.pth")
                        torch.save(best_checkpoint, best_model_path)
                        print(f"New best model saved with combined macro recall: {best_combined_recall:.4f}")
                    else:
                        patience_counter += 1
                        print(f"No improvement in combined macro recall. Patience: {patience_counter}/{early_stopping_patience}")
                else:
                    # For non-curriculum training, use validation loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model checkpoint
                        best_checkpoint = checkpoint.copy()
                        best_checkpoint['best_val_loss'] = best_val_loss
                        
                        best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_curriculum.pth")
                        torch.save(best_checkpoint, best_model_path)
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
