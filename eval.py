import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MVFoulsModel
from dataset import MVFoulsDataset, custom_collate_fn
import os
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix # Import metrics
from transform import get_val_transforms # Import val transform
from torchvision.models.video import MViT_V2_S_Weights # Needed to get model input size
import glob
from natsort import natsorted
from decord import VideoReader, cpu
import numpy as np

def predict_unannotated_dataset(model_path, data_folder="mvfouls", split="challenge", start_frame=63, end_frame=86):
    """
    Predict on unannotated datasets (like challenge set) that don't have annotations.json
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    MODEL_INPUT_SIZE = (224, 224)

    # Load the model
    model = MVFoulsModel().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using randomly initialized model.")
    model.eval()

    # Get video transforms
    transform = get_val_transforms(MODEL_INPUT_SIZE)

    # Action and Severity label maps
    action_labels_map = {
        0: "Tackling",
        1: "Standing tackling", 
        2: "High leg",
        3: "Holding",
        4: "Pushing",
        5: "Elbowing",
        6: "Challenge",
        7: "Dive"
    }

    offence_labels_map = {
        0: "No offence",
        1: "Offence",
        2: "Offence", 
        3: "Offence"
    }

    severity_numerical_map = {
        0: "",
        1: "1.0",
        2: "3.0", 
        3: "5.0"
    }

    # Find all action directories in the split
    split_path = os.path.join(data_folder, split)
    if not os.path.exists(split_path):
        print(f"Error: Split directory not found at {split_path}")
        return

    action_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    action_dirs = natsorted(action_dirs)
    
    # Extract numeric action IDs from directory names
    action_numbers = []
    for action_dir in action_dirs:
        # Extract number from directory name (e.g., "action_001" -> "1", "1" -> "1")
        import re
        match = re.search(r'(\d+)', action_dir)
        if match:
            action_numbers.append(str(int(match.group(1))))  # Remove leading zeros
        else:
            action_numbers.append(action_dir)  # Fallback to original name
    
    print(f"Found {len(action_dirs)} action directories in {split} split")

    results = {
        "Set": split,
        "Actions": {}
    }

    # Use all frames without downsampling
    duration_frames = end_frame - start_frame + 1

    print(f"Frame sampling config: start={start_frame}, end={end_frame}")
    print(f"Duration: {duration_frames} frames, intelligently downsampled to 16 frames for MViT")
    print(f"Output frames: 16 (uniform sampling from {duration_frames} frames)")

    with torch.no_grad():
        for i, action_dir in enumerate(tqdm(action_dirs, desc=f"Predicting {split}")):
            action_path = os.path.join(split_path, action_dir)
            action_number = action_numbers[i]  # Get the corresponding numeric ID
            
            # Find all video files in this action directory
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(os.path.join(action_path, ext)))
            
            if not video_files:
                print(f"Warning: No video files found in {action_path}")
                continue

            video_files = natsorted(video_files)
            
            # Load and process all clips for this action
            clips_tensors = []
            
            for video_file in video_files:
                try:
                    # Load video using decord
                    vr = VideoReader(video_file, ctx=cpu(0))
                    total_frames = len(vr)
                    
                    # Get all frames in the range first (same as dataset class)
                    all_frame_indices = list(range(start_frame, end_frame + 1))  # 63 to 86 inclusive = 24 frames
                    
                    # Handle videos shorter than expected
                    if end_frame >= total_frames:
                        all_frame_indices = list(range(min(start_frame, total_frames-1), total_frames))
                    
                    if not all_frame_indices:
                        continue
                    
                    # Load all frames in the range
                    all_frames = vr.get_batch(all_frame_indices).asnumpy()  # Shape: (T, H, W, C)
                    
                    # Intelligent temporal downsampling to exactly 16 frames for MViT (same as dataset class)
                    if len(all_frames) > 16:
                        # Use uniform sampling to get exactly 16 frames
                        indices = np.linspace(0, len(all_frames) - 1, 16, dtype=int)
                        selected_frames = [all_frames[i] for i in indices]
                    else:
                        # If we have 16 or fewer frames, use all and pad if necessary
                        selected_frames = list(all_frames)
                        while len(selected_frames) < 16:
                            selected_frames.append(selected_frames[-1])  # Repeat last frame
                    
                    # Convert to tensor (exactly same as dataset class)
                    video = torch.from_numpy(np.stack(selected_frames))
                    
                    # Apply transforms (same as dataset class)
                    if transform:
                        try:
                            video = transform(video)  # This handles all preprocessing
                        except Exception as transform_error:
                            print(f"Transform error for {video_file}: {transform_error}")
                            continue
                    
                    clips_tensors.append(video)
                    
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    continue
            
            if not clips_tensors:
                print(f"Warning: No valid clips found for action {action_number}")
                continue
            
            # Stack clips into a single tensor (same as dataset class)
            # Shape: (num_clips_for_action, C, num_frames, H, W)
            combined_videos = torch.stack(clips_tensors).to(DEVICE)
            
            # Forward pass - model expects a list with one tensor per action
            action_logits, severity_logits = model([combined_videos])
            
            # Get predictions
            _, predicted_action_idx = torch.max(action_logits, 1)
            _, predicted_severity_idx = torch.max(severity_logits, 1)
            
            # Convert to human-readable labels
            predicted_action_class = action_labels_map.get(predicted_action_idx.item(), "Unknown Action")
            predicted_offence = offence_labels_map.get(predicted_severity_idx.item(), "Unknown Offence")
            predicted_severity_value = severity_numerical_map.get(predicted_severity_idx.item(), "")
            
            # Store results
            results["Actions"][action_number] = {
                "Action class": predicted_action_class,
                "Offence": predicted_offence,
                "Severity": predicted_severity_value
            }

    # Save results to JSON file
    output_filename = f"predictions_{split}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {output_filename}")
    
    # Print summary
    print(f"\nPrediction Summary for {split}:")
    print(f"Total actions processed: {len(results['Actions'])}")
    
    # Count predictions by class
    action_counts = {}
    severity_counts = {}
    
    for action_data in results["Actions"].values():
        action_class = action_data["Action class"]
        severity = action_data["Severity"]
        
        action_counts[action_class] = action_counts.get(action_class, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("\nAction class distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")
    
    print("\nSeverity distribution:")
    for severity, count in sorted(severity_counts.items()):
        severity_name = "No Offence" if severity == "" else f"Severity {severity}"
        print(f"  {severity_name}: {count}")

def evaluate_model(model_path, data_folder="mvfouls", test_split="test", start_frame=63, end_frame=86):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    MODEL_INPUT_SIZE = (224, 224) # Official MViT input size after preprocessing

    # Load the model
    model = MVFoulsModel().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using randomly initialized model.")
    model.eval() # Set model to evaluation mode

    # Prepare dataset and dataloader
    try:
        test_dataset = MVFoulsDataset(data_folder, test_split, start_frame, end_frame, transform_model=get_val_transforms(MODEL_INPUT_SIZE), target_fps=17)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn) # Batch size 1 for individual predictions
        print(f"Test dataset initialized with {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error initializing test dataset or dataloader: {e}")
        return

    if len(test_dataset) == 0:
        print("No samples found in the test dataset. Exiting evaluation.")
        return

    results = {
        "Set": test_split,
        "Actions": {}
    }

    # Action and Severity label maps (should match data_loader.py or dataset.py)
    action_labels_map = {
        0: "Tackling",
        1: "Standing tackling",
        2: "High leg",
        3: "Holding",
        4: "Pushing",
        5: "Elbowing",
        6: "Challenge",
        7: "Dive"
    }
    severity_labels_map = {
        0: "No Offence", # Not requested in output, but good for completeness
        1: "Offence + No Card",
        2: "Offence + Yellow Card",
        3: "Offence + Red Card"
    }

    offence_labels_map = {
        0: "No offence",
        1: "Offence",
        2: "Offence",
        3: "Offence"
    }

    severity_numerical_map = {
        0: "",
        1: "1.0",
        2: "3.0",
        3: "5.0"
    }

    all_action_labels = []
    all_predicted_actions = []
    all_severity_labels = []
    all_predicted_severities = []

    with torch.no_grad():
        for i, (videos, action_labels, severity_labels, action_ids) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Move data to device
            # videos is now a list of tensors, where each tensor corresponds to an action event
            videos = [v.to(DEVICE) for v in videos]
            
            # Convert one-hot encoded labels to class indices
            # Labels are now (batch_size, num_classes) directly from custom_collate_fn
            action_labels_idx = torch.argmax(action_labels, dim=1).to(DEVICE)
            severity_labels_idx = torch.argmax(severity_labels, dim=1).to(DEVICE)

            # Forward pass
            action_logits, severity_logits = model(videos)

            # Get predictions
            _, predicted_action_idx = torch.max(action_logits, 1)
            _, predicted_severity_idx = torch.max(severity_logits, 1)

            # Collect predictions and true labels for metrics
            all_action_labels.extend(action_labels_idx.cpu().numpy())
            all_predicted_actions.extend(predicted_action_idx.cpu().numpy())
            all_severity_labels.extend(severity_labels_idx.cpu().numpy())
            all_predicted_severities.extend(predicted_severity_idx.cpu().numpy())

            # Map to human-readable labels for JSON output
            # Loop through the batch of predictions and action_ids
            for batch_idx in range(len(action_ids)): # Iterate over the batch size (number of actions)
                current_action_id = action_ids[batch_idx]
                current_predicted_action_idx = predicted_action_idx[batch_idx].item()
                current_predicted_severity_idx = predicted_severity_idx[batch_idx].item()

                predicted_action_class = action_labels_map.get(current_predicted_action_idx, "Unknown Action")
                predicted_offence = offence_labels_map.get(current_predicted_severity_idx, "Unknown Offence")
                predicted_severity_value = severity_numerical_map.get(current_predicted_severity_idx, "")

                # Populate results dictionary
                results["Actions"][str(current_action_id)] = {
                    "Action class": predicted_action_class,
                    "Offence": predicted_offence,
                    "Severity": predicted_severity_value
                }

    # Save results to JSON file
    output_filename = f"evaluation_results_{test_split}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {output_filename}")

    # Calculate and print metrics
    if len(all_action_labels) > 0:
        # Action Classification Metrics
        action_accuracy = accuracy_score(all_action_labels, all_predicted_actions)
        _, action_recall, _, _ = precision_recall_fscore_support(all_action_labels, all_predicted_actions, average='macro', zero_division=0)

        print(f"\nAction Classification Metrics:")
        print(f"  Accuracy: {action_accuracy:.4f}")
        print(f"  Macro Recall: {action_recall:.4f}")
        
        # Action Confusion Matrix
        action_cm = confusion_matrix(all_action_labels, all_predicted_actions)
        print(f"  Confusion Matrix (Actions):\n{action_cm}")

        # Per-class Action Metrics
        action_precision_per_class, action_recall_per_class, action_f1_per_class, _ = precision_recall_fscore_support(all_action_labels, all_predicted_actions, average=None, zero_division=0)
        print("\nPer-class Action Metrics:")
        for i, class_name in action_labels_map.items():
            print(f"  {class_name}: Precision={action_precision_per_class[i]:.4f}, Recall={action_recall_per_class[i]:.4f}, F1-Score={action_f1_per_class[i]:.4f}")

        # Severity Classification Metrics
        severity_accuracy = accuracy_score(all_severity_labels, all_predicted_severities)
        _, severity_recall, _, _ = precision_recall_fscore_support(all_severity_labels, all_predicted_severities, average='macro', zero_division=0)

        print(f"\nSeverity Classification Metrics:")
        print(f"  Accuracy: {severity_accuracy:.4f}")
        print(f"  Macro Recall: {severity_recall:.4f}")
        
        # Severity Confusion Matrix
        severity_cm = confusion_matrix(all_severity_labels, all_predicted_severities)
        print(f"  Confusion Matrix (Severity):\n{severity_cm}")

        # Per-class Severity Metrics
        severity_precision_per_class, severity_recall_per_class, severity_f1_per_class, _ = precision_recall_fscore_support(all_severity_labels, all_predicted_severities, average=None, zero_division=0)
        print("\nPer-class Severity Metrics:")
        for i, class_name in severity_labels_map.items():
            print(f"  {class_name}: Precision={severity_precision_per_class[i]:.4f}, Recall={severity_recall_per_class[i]:.4f}, F1-Score={severity_f1_per_class[i]:.4f}")
    else:
        print("No samples processed for metrics calculation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MVFoulsModel on a dataset or predict on unannotated data.")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the pre-trained model checkpoint (.pth file).')
    parser.add_argument('--data_folder', type=str, default='mvfouls', 
                        help='Path to the dataset folder (e.g., "mvfouls").')
    parser.add_argument('--split', type=str, default='test', 
                        help='Name of the dataset split to evaluate/predict on (e.g., "test", "challenge").')
    parser.add_argument('--start_frame', type=int, default=63, 
                        help='Start frame for video clips.')
    parser.add_argument('--end_frame', type=int, default=86, 
                        help='End frame for video clips.')
    parser.add_argument('--predict_only', action='store_true',
                        help='Use this flag for unannotated datasets (like challenge) that only need predictions.')
    
    args = parser.parse_args()

    # Check if annotations.json exists to determine which function to use
    annotations_path = os.path.join(args.data_folder, args.split, "annotations.json")
    
    if args.predict_only or not os.path.exists(annotations_path):
        print(f"No annotations found or --predict_only flag used. Running prediction mode for {args.split} split.")
        predict_unannotated_dataset(
            model_path=args.model_path,
            data_folder=args.data_folder,
            split=args.split,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
    else:
        print(f"Annotations found. Running evaluation mode for {args.split} split.")
        evaluate_model(
            model_path=args.model_path,
            data_folder=args.data_folder,
            test_split=args.split,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
