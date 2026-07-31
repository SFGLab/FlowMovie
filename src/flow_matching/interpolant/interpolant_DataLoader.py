import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.trajectories = data["trajectories"]
        self.metadata = data["metadata"]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Shape: (n, K*3+2)
        traj = self.trajectories[idx]
        
        # Ensure float32 for training
        if traj.dtype != torch.float32:
            traj = traj.to(torch.float32)
            
        return traj

    def get_N(self):
        return self.metadata["num_trajectories"]

    def get_n(self):
        return self.metadata["num_frames"]

    def get_num_beads(self):
        return self.metadata["num_beads"]


if __name__ == "__main__":
    # Example Loading
    test_name = "test02"
    phase_trans = "G2M_to_G1"
    
    dataset = TrajectoryDataset(f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)