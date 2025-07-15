from torch.utils.data import Dataset
import torch
from decord import VideoReader, cpu
from data_loader import load_clips, label_to_numerical
from natsort import natsorted
from torchvision.models.video import MViT_V2_S_Weights

class MVFoulsDataset(Dataset):
    def __init__(self, folder_path, split, start_frame, end_frame, transform=None, transform_model=None):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.split = split
        self.transform = transform
        self.transform_model = transform_model
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
            frame_indices = range(self.start_frame, self.end_frame + 1)  
            video = vr.get_batch(frame_indices).asnumpy()
            
            # Convert numpy array to torch tensor with (T, H, W, C) format first for torchvision transforms
            video = torch.from_numpy(video)
            
    
            video = self.transform_model(video) # Apply the passed-in transform
            
            all_clips_for_action_data.append(video)
        # the shape for combined_videos is (num_clips_for_this_action, C, num_frames, H, W)
        combined_videos = torch.stack(all_clips_for_action_data) # Stack all clips into a single tensor
        return combined_videos, labels_action, labels_severity, action_id

def custom_collate_fn(batch):
    # 'batch' is a list of tuples: [(video_0, action_label_0, severity_label_0), (video_1, action_label_1, severity_label_1), ...]
    # where video_i has shape (num_clips_for_action_i, C, num_frames, H, W)

    # Flatten the list of video tensors from (num_clips_for_action_i, C, num_frames, H, W) to (total_clips_in_batch, C, num_frames, H, W)
    videos = torch.cat([item[0] for item in batch], dim=0) # Concatenate all clips from all samples in the batch

    # Expand labels to match the number of clips for each action
    # Each item[0] is combined_videos with shape (num_clips_for_action_i, ...)
    # We need to repeat action_label_i and severity_label_i 'num_clips_for_action_i' times
    action_labels = []
    severity_labels = []
    action_ids = [] # This is now tricky, as action_id is per action, not per clip

    for item in batch:
        num_clips = item[0].shape[0] # Number of clips for the current action
        action_labels.append(item[1].repeat(num_clips, 1)) # Repeat action label for each clip
        severity_labels.append(item[2].repeat(num_clips, 1)) # Repeat severity label for each clip
        action_ids.extend([item[3]] * num_clips) # Repeat action ID for each clip

    action_labels = torch.cat(action_labels, dim=0) # Concatenate all action labels
    severity_labels = torch.cat(severity_labels, dim=0) # Concatenate all severity labels

    return videos, action_labels, severity_labels, action_ids


if __name__ == "__main__":
    # Test configuration
    test_folder = "mvfouls"
    test_split = "train"
    start_frame = 67
    end_frame = 82
    
    # Initialize dataset
    dataset = MVFoulsDataset(test_folder, test_split, start_frame, end_frame)
    
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
        print(f"Batch action IDs shape: {batch_action_ids.shape}")
        
        print("\nCustom collate function test passed!")
    else:
        print("No samples found - ensure dataset is downloaded to 'mvfouls' folder")