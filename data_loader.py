import os
import json
import torch
import glob
from natsort import natsorted

# First label: {No Offence, Offence + No Card, Offence + Yellow Card, Offence + Red Card} 
# Second label: {Standing Tackle, Tackle, Holding, Pushing, Challenge, Dive, High Leg, Elbowing}
def label_to_numerical(folder_path, split):
    path_annotations = os.path.join(folder_path, split, "annotations.json")
    dictionary_action =  {"Tackling":0,"Standing tackling":1,"High leg":2,"Holding":3,"Pushing":4,
                        "Elbowing":5, "Challenge":6, "Dive":7, "Dont know":8}
    inverse_dictionary_action = {v: k for k, v in dictionary_action.items()}
    
    if not os.path.exists(path_annotations):
        raise FileNotFoundError(f"Annotations file not found at {path_annotations}")
    else:
        with open(path_annotations, 'r') as f:
            annotations = json.load(f)
            
    num_severity_classes = 4
    num_action_classes = 8
    
    useless_actions = []
    labels_action = []
    labels_severity = []
    
    for action in annotations['Actions']:
        action_class = annotations['Actions'][action]['Action class']
        severity_class = annotations['Actions'][action]['Severity']
        offence_class = annotations['Actions'][action]['Offence']
        
        if action_class == '' or action_class == 'Dont know':
            useless_actions.append(action)
            continue

        if (offence_class == '' or offence_class == 'Between') and action_class != 'Dive':
            useless_actions.append(action)
            continue

        if (severity_class == '' or severity_class == '2.0' or severity_class == '4.0') and action_class != 'Dive' and offence_class != 'No offence' and offence_class != 'No Offence':
            useless_actions.append(action)
            continue
        
        if offence_class == '' or offence_class == 'Between':
            offence_class = 'Offence'
        
        if severity_class == '' or severity_class == '2.0' or severity_class == '4.0':
            severity_class = '1.0'
        
        # FIRST LABEL ONE HOT ENCODING
        # NO OFFENCE
        if offence_class == 'No offence' or offence_class == 'No Offence':
            labels_severity.append(torch.zeros(1, num_severity_classes))
            labels_severity[len(labels_severity) - 1][0][0] = 1
        # OFFENCE + NO CARD
        elif offence_class == 'Offence' and severity_class == '1.0':
            labels_severity.append(torch.zeros(1, num_severity_classes))
            labels_severity[len(labels_severity) - 1][0][1] = 1
        # OFFENCE + YELLOW CARD
        elif offence_class == 'Offence' and severity_class == '3.0':
            labels_severity.append(torch.zeros(1, num_severity_classes))
            labels_severity[len(labels_severity) - 1][0][2] = 1
        # OFFENCE + RED CARD
        elif offence_class == 'Offence' and severity_class == '5.0':
            labels_severity.append(torch.zeros(1, num_severity_classes))
            labels_severity[len(labels_severity) - 1][0][3] = 1
        else:
            useless_actions.append(action)
            continue
        
        # SECOND LABEL ONE HOT ENCODING
        labels_action.append(torch.zeros(1, num_action_classes))
        labels_action[len(labels_action) - 1][0][dictionary_action[action_class]] = 1
        
        return labels_action, labels_severity, useless_actions
    


def load_clips(folder_path, split, usless_actions):
    """Load all clip paths for each action in the dataset, excluding useless actions.
    
    Args:
        folder_path: Root folder containing dataset splits
        split: Dataset split (e.g. 'train', 'val')
        
    Returns:
        Dictionary mapping action names to lists of clip paths
    """
    
    clips_dict = {}
    split_path = os.path.join(folder_path, split)
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split directory not found at {split_path}")
        
    # Find all action directories and sort numerically
    action_dirs = natsorted(glob.glob(os.path.join(split_path, "action_*")))
    
    for action_dir in action_dirs:
        action_name = os.path.basename(action_dir)
        
        # Skip useless actions
        if action_name in useless_actions:
            continue
            
        # Find all mp4 files in action directory
        clip_paths = glob.glob(os.path.join(action_dir, "clip_*.mp4"))
        clips_dict[action_name] = sorted(clip_paths)
        
    return clips_dict


# Example usage
if __name__ == "__main__":
    # Label to numerical conversion example
    folder_path = 'mvfouls'
    split = 'train'
    labels_action, labels_severity, useless_actions = label_to_numerical(folder_path, split)
    print("Action Labels:", labels_action)
    print("Severity Labels:", labels_severity)
    print("Useless Actions:", useless_actions)
    
    # Clip loading example - show first 5 action-clip pairs
    clips = load_clips(folder_path, split, useless_actions)
    print("\nFirst 5 action-clip pairs:")
    for i, (action, clip_paths) in enumerate(clips.items()):
        if i >= 5:
            break
        print(f"{action}: {clip_paths}")
