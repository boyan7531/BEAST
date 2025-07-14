
import os
from collections import Counter
import torch
from data_loader import label_to_numerical

def analyze_label_distribution(folder_path="mvfouls", splits=["train", "valid"]):
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
        0: "No Offence",
        1: "Offence + No Card",
        2: "Offence + Yellow Card",
        3: "Offence + Red Card"
    }

    for split in splits:
        print(f"\n--- Analyzing {split.upper()} Set ---\n")
        try:
            labels_action_one_hot, labels_severity_one_hot, useless_actions = label_to_numerical(folder_path, split)
        except FileNotFoundError as e:
            print(f"Error loading data for {split}: {e}")
            continue

        action_counts = Counter()
        severity_counts = Counter()

        # Convert one-hot to integer labels and count
        for label_tensor in labels_action_one_hot:
            action_idx = torch.argmax(label_tensor).item()
            action_counts[action_labels_map[action_idx]] += 1
        
        for label_tensor in labels_severity_one_hot:
            severity_idx = torch.argmax(label_tensor).item()
            severity_counts[severity_labels_map[severity_idx]] += 1

        print("Action Label Distribution:")
        total_action_labels = sum(action_counts.values())
        for label, count in action_counts.most_common():
            percentage = (count / total_action_labels) * 100 if total_action_labels > 0 else 0
            print(f"- {label}: {count} ({percentage:.2f}%)")

        print("\nSeverity Label Distribution:")
        total_severity_labels = sum(severity_counts.values())
        for label, count in severity_counts.most_common():
            percentage = (count / total_severity_labels) * 100 if total_severity_labels > 0 else 0
            print(f"- {label}: {count} ({percentage:.2f}%)")

        if useless_actions:
            print(f"\nNumber of useless actions (excluded): {len(useless_actions)}")

if __name__ == "__main__":
    analyze_label_distribution()

