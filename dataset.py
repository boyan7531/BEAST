from torch.utils.data import Dataset
import torch
import numpy as np
from decord import VideoReader, cpu
from data_loader import load_clips, label_to_numerical
from natsort import natsorted
from torchvision.models.video import MViT_V2_S_Weights

class MVFoulsDataset(Dataset):
    def __init__(self, folder_path, split, start_frame, end_frame, transform=None, transform_model=None, target_fps=17):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.split = split
        self.transform = transform
        self.transform_model = transform_model
        self.target_fps = target_fps
        
        # Use all frames without downsampling
        duration_frames = end_frame - start_frame + 1  # +1 for inclusive range (63 to 86 = 24 frames)
        
        print(f"Frame sampling config: start={start_frame}, end={end_frame}")
        print(f"Duration: {duration_frames} frames, intelligently downsampled to 16 frames for MViT")
        print(f"Output frames: 16 (uniform sampling from {duration_frames} frames)")
        self.labels_action_list, self.labels_severity_list, self.useless_actions = label_to_numerical(folder_path, split)
        self.clip_paths_dict = load_clips(folder_path, split, self.useless_actions)
        self.length = len(self.labels_action_list)
        self.data_list = []
        useful_actions = natsorted(self.clip_paths_dict.keys())
        print("useful_actions:", len(useful_actions))
        print("labels_action_list:", len(self.labels_action_list))
        print("labels_severity_list:", len(self.labels_severity_list))
        if not (len(useful_actions) == len(self.labels_action_list) == len(self.labels_severity_list)):
            print("they are not the same size")
        for action_name, labels_action, labels_severity in zip(useful_actions, self.labels_action_list, self.labels_severity_list):
            # Append each clip path for this action along with its labels
            self.data_list.append((self.clip_paths_dict[action_name], labels_action, labels_severity, action_name)) # Append action_name instead of clip_path

        self.length = len(self.data_list)


        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        all_clips_for_action_paths, labels_action, labels_severity, action_name = self.data_list[idx]
        
        # Extract numerical action ID from action_name (e.g., "action_0" -> "0")
        action_id = action_name.split('_')[1]
        
        all_clips_for_action_data = []
        for clip_path in all_clips_for_action_paths:
            vr = VideoReader(clip_path, ctx=cpu(0))
            
            # Get all frames in the range first (63 to 86 inclusive = 24 frames)
            all_frame_indices = list(range(self.start_frame, self.end_frame + 1))  # 63 to 86 inclusive = 24 frames
            all_frames = vr.get_batch(all_frame_indices).asnumpy()
            
            # Intelligent temporal downsampling to exactly 16 frames for MViT
            if len(all_frames) > 16:
                # Use uniform sampling to get exactly 16 frames
                indices = torch.linspace(0, len(all_frames) - 1, 16).long()
                selected_frames = [all_frames[i] for i in indices]
            else:
                # If we have 16 or fewer frames, use all and pad if necessary
                selected_frames = all_frames
                while len(selected_frames) < 16:
                    selected_frames.append(selected_frames[-1])  # Repeat last frame
            
            # Convert to tensor
            video = torch.from_numpy(np.stack(selected_frames))
            
            video = self.transform_model(video) # Apply the passed-in transform
            
            all_clips_for_action_data.append(video)
        # the shape for combined_videos is (num_clips_for_this_action, C, num_frames, H, W)
        combined_videos = torch.stack(all_clips_for_action_data) # Stack all clips into a single tensor
        return combined_videos, labels_action, labels_severity, action_id

def custom_collate_fn(batch):
    # 'batch' is a list of tuples: [(video_0, action_label_0, severity_label_0, action_id_0), ...]
    # where video_i has shape (num_clips_for_action_i, C, num_frames, H, W)

    # Keep videos as a list of tensors, where each tensor corresponds to one action event
    videos = [item[0] for item in batch]

    # action_labels and severity_labels are already (1, num_classes) from dataset.__getitem__
    # Concatenate them to form (batch_size, num_classes) where batch_size is the number of actions
    action_labels = torch.cat([item[1] for item in batch], dim=0)
    severity_labels = torch.cat([item[2] for item in batch], dim=0)

    # Keep action_ids as a list of strings, one per action event
    action_ids = [item[3] for item in batch]

    return videos, action_labels, severity_labels, action_ids


if __name__ == "__main__":
    # Test configuration
    test_folder = "mvfouls"
    test_split = "train"
    start_frame = 63
    end_frame = 86
    
    # Initialize dataset
    dataset = MVFoulsDataset(test_folder, test_split, start_frame, end_frame, target_fps=17)
    
    # Test custom collate function with a small batch
    from torch.utils.data import DataLoader
    
    # Create a DataLoader with the custom collate_fn
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=custom_collate_fn)
    
    # Get a batch
    if len(dataset) > 0:
        batch_videos, batch_action_labels, batch_severity_labels, batch_action_ids = next(iter(dataloader))
        
        print(f"\nBatch of videos is a list of {len(batch_videos)} tensors.")
        for i, video_tensor in enumerate(batch_videos):
            print(f"  Video {i} shape: {video_tensor.shape}")
        
        print(f"Batch action labels shape: {batch_action_labels.shape}")
        print(f"Batch severity labels shape: {batch_severity_labels.shape}")
        # For action_ids, we expect a list of strings, so checking length is more appropriate than .shape
        print(f"Batch action IDs count: {len(batch_action_ids)}")
        print(f"First few action IDs: {batch_action_ids[:min(3, len(batch_action_ids))]}...")
        
        print("\nCustom collate function test passed!")
    else:
        print("No samples found - ensure dataset is downloaded to 'mvfouls' folder")