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
            self.data_list.append((self.clip_paths_dict[action_name], labels_action, labels_severity)) # Append action_name instead of clip_path

        self.length = len(self.data_list)

        # Initialize the model transform from MViT_V2_S_Weights
        # We only need the transforms, not the model itself for preprocessing
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.transform_model = weights.transforms()

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        all_clips_for_action_paths, labels_action, labels_severity = self.data_list[idx]
        
        all_clips_for_action_data = []
        for clip_path in all_clips_for_action_paths:
            vr = VideoReader(clip_path, ctx=cpu(0))
            frame_indices = range(self.start_frame, self.end_frame + 1)  
            video = vr.get_batch(frame_indices).asnumpy()
            
            # Convert numpy array to torch tensor with (T, H, W, C) format first for torchvision transforms
            video = torch.from_numpy(video)
            
            # Apply the model's expected transforms
            # The transform expects (T, C, H, W) or (B, T, C, H, W) and outputs (C, T, H, W) for single video
            # We provide (T, H, W, C) and then permute to (T, C, H, W) before applying transform
            video = video.permute(0, 3, 1, 2) # Change from (T, H, W, C) to (T, C, H, W)
            video = self.transform_model(video)
            
            all_clips_for_action_data.append(video)
        # the shape for combined_videos is (num_clips_for_this_action, C, num_frames, H, W)
        combined_videos = torch.stack(all_clips_for_action_data) # Stack all clips into a single tensor
        return combined_videos, labels_action, labels_severity

def custom_collate_fn(batch):
    # 'batch' is a list of tuples: [(video_0, action_label_0, severity_label_0), (video_1, action_label_1, severity_label_1), ...]
    # where video_i has shape (num_clips_for_action_i, C, num_frames, H, W)

    videos = [item[0] for item in batch] # This will be the list of tensors for the model's forward pass
    action_labels = torch.cat([item[1] for item in batch], dim=0) # Concatenate action labels
    severity_labels = torch.cat([item[2] for item in batch], dim=0) # Concatenate severity labels

    return videos, action_labels, severity_labels


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
        batch_videos, batch_action_labels, batch_severity_labels = next(iter(dataloader))
        
        print(f"\nBatch of videos is a list of {len(batch_videos)} tensors.")
        for i, video_tensor in enumerate(batch_videos):
            print(f"  Video {i} shape: {video_tensor.shape}")
        
        print(f"Batch action labels shape: {batch_action_labels.shape}")
        print(f"Batch severity labels shape: {batch_severity_labels.shape}")
        
        print("\nCustom collate function test passed!")
    else:
        print("No samples found - ensure dataset is downloaded to 'mvfouls' folder")