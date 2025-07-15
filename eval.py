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

def evaluate_model(model_path, data_folder="mvfouls", test_split="test", start_frame=67, end_frame=82):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    MODEL_INPUT_SIZE = (224, 224) # MViT models typically use 224x224 input resolution

    # Load the model
    model = MVFoulsModel().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using randomly initialized model.")
    model.eval() # Set model to evaluation mode

    # Prepare dataset and dataloader
    try:
        test_dataset = MVFoulsDataset(data_folder, test_split, start_frame, end_frame, transform_model=get_val_transforms(MODEL_INPUT_SIZE))
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
            videos = videos.to(DEVICE)
            
            # Convert one-hot encoded labels to class indices
            action_labels_idx = torch.argmax(action_labels.squeeze(1), dim=1).to(DEVICE)
            severity_labels_idx = torch.argmax(severity_labels.squeeze(1), dim=1).to(DEVICE)

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
            predicted_action_class = action_labels_map.get(predicted_action_idx[0].item(), "Unknown Action")
            
            predicted_offence = offence_labels_map.get(predicted_severity_idx[0].item(), "Unknown Offence")
            predicted_severity_value = severity_numerical_map.get(predicted_severity_idx[0].item(), "")

            # Populate results dictionary
            for idx, action_id in enumerate(action_ids):
                results["Actions"][str(action_id)] = {
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
    parser = argparse.ArgumentParser(description="Evaluate MVFoulsModel on a test dataset.")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the pre-trained model checkpoint (.pth file).')
    parser.add_argument('--data_folder', type=str, default='mvfouls', 
                        help='Path to the dataset folder (e.g., "mvfouls").')
    parser.add_argument('--test_split', type=str, default='test', 
                        help='Name of the dataset split to evaluate on (e.g., "test").')
    parser.add_argument('--start_frame', type=int, default=67, 
                        help='Start frame for video clips.')
    parser.add_argument('--end_frame', type=int, default=82, 
                        help='End frame for video clips.')
    
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_folder=args.data_folder,
        test_split=args.test_split,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
