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
            for clip_path in self.clip_paths_dict[action_name]:
                self.data_list.append((clip_path, labels_action, labels_severity))

        self.length = len(self.data_list)

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        clip_path, labels_action, labels_severity = self.data_list[idx]
        vr = VideoReader(clip_path, ctx=cpu(0))
        frame_indices = range(self.start_frame, self.end_frame + 1)  
        video = vr.get_batch(frame_indices).asnumpy()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  
        return video, labels_action, labels_severity, clip_path


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
        print(f"\nLoaded video shape: {video.shape} (Expecting: (16, 3, H, W))")
        print(f"Number of frames: {video.size(0)} (Expected: {end_frame - start_frame + 1})")
        
        # Frame index validation
        vr = VideoReader(path, ctx=cpu(0))
        try:
            first_frame = vr[start_frame].asnumpy()
            last_frame = vr[end_frame].asnumpy()
            
            # Convert to tensor and permute for comparison
            first_tensor = torch.from_numpy(first_frame).permute(2, 0, 1)
            last_tensor = torch.from_numpy(last_frame).permute(2, 0, 1)
            
            # Verify frame accuracy
            assert torch.allclose(video[0], first_tensor, atol=1e-4), "First frame mismatch"
            assert torch.allclose(video[-1], last_tensor, atol=1e-4), "Last frame mismatch"
            print("Frame indices validated successfully")
            
        except IndexError as e:
            print(f"Frame index error: {e}. Video has {len(vr)} frames.")
            
        print("\nBasic functionality tests passed!")
    else:
        print("No samples found - ensure dataset is downloaded to 'mvfouls' folder")