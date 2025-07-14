from torch.utils.data import Dataset
import torch
from decord import VideoReader, cpu
from data_loader import load_clips, label_to_numerical
from natsort import natsorted

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

        print(self.data_list[0])
        print(self.data_list[1])
        print(self.data_list[2])

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        all_clips_for_action_paths, labels_action, labels_severity = self.data_list[idx]
        
        all_clips_for_action_data = []
        for clip_path in all_clips_for_action_paths:
            vr = VideoReader(clip_path, ctx=cpu(0))
            frame_indices = range(self.start_frame, self.end_frame + 1)  
            video = vr.get_batch(frame_indices).asnumpy()
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            all_clips_for_action_data.append(video)
        
        combined_videos = torch.stack(all_clips_for_action_data) # Stack all clips into a single tensor
        return combined_videos, labels_action, labels_severity, all_clips_for_action_paths # Return action_name instead of clip_path


if __name__ == "__main__":
    # Test configuration
    test_folder = "mvfouls"
    test_split = "train"
    start_frame = 67
    end_frame = 82
    
    # Initialize dataset
    dataset = MVFoulsDataset(test_folder, test_split, start_frame, end_frame)
    
    # Get first sample
    
    if len(dataset) > 0:
        video, action, severity, path = dataset[0]
        
        # Basic checks
        # The shape of video will now be (num_clips_in_action, frames_per_clip, C, H, W)
        print(f"\nLoaded video shape: {video.shape} (Expecting: (num_clips, {end_frame - start_frame + 1}, 3, H, W))")
        print(f"Number of clips in action: {video.size(0)}")
        print("\nBasic functionality tests passed!")
    else:
        print("No samples found - ensure dataset is downloaded to 'mvfouls' folder")