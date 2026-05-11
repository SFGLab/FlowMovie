import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_cif(cif_path):
    """
    Reads the cif file from path.
    Returns the np.array with (K, 3) dimensions with 3D positions of K atoms from the file. 
    """
    position_arrays = []
    with open(cif_path, 'r') as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                line_split = line.split()
                x, y, z = line_split[10], line_split[11], line_split[12]
                position_arrays.append(np.array([x, y, z], dtype=float))
            elif line.startswith("C0 covale"):
                break
    return np.hstack(position_arrays)
            

def preprocess_chromatin_trajectories(root_dir, output_file, atoms_per_structure=None):
    """
    root_dir: Path to the folder containing N subfolders (trajectories)
    output_file: Path to save the processed .pt file
    atoms_per_structure: (Optional) Expected K atoms to verify consistency
    """
    all_trajectories = []
    root_path = Path(root_dir)
    
    # Identify subfolders (N trajectories)
    traj_folders = sorted([f for f in root_path.iterdir() if f.is_dir()])
    
    print(f"Found {len(traj_folders)} trajectories. Starting processing...")

    for folder in tqdm(traj_folders, desc="Processing Trajectories"):
        # Identify all .cif files in chronological/ordered order (M structures)
        cif_files = sorted([f for f in os.listdir(os.path.join(folder, 'frames_cif')) if f.endswith('.cif')])
        max_step = max([int(f.split('_')[0][4:]) for f in cif_files])
        cif_files = [f for f in cif_files if int(f.split('_')[0][4:]) == max_step] # only preprocesses the last step
        cif_files = [os.path.join(folder, 'frames_cif', f) for f in cif_files]
        
        trajectory_coords = []
        
        for cif_file in cif_files:
            try:
                # Extract (K, 3) coordinates
                coords = read_cif(cif_file)
                
                if atoms_per_structure and coords.shape[0] != atoms_per_structure:
                    print(f"Warning: {cif_file} has {coords.shape[0]} atoms, expected {atoms_per_structure}")
                
                trajectory_coords.append(coords)
            except Exception as e:
                print(f"Error reading {cif_file}: {e}")

        # Convert list of arrays to a single tensor (M, K, 3)
        if trajectory_coords:
            traj_tensor = torch.tensor(np.array(trajectory_coords), dtype=torch.float32)
            all_trajectories.append(traj_tensor)

    # Save as a dictionary for easy loading
    data_to_save = {
        "trajectories": all_trajectories, # List of (M, K, 3) tensors
        "metadata": {
            "num_trajectories": len(all_trajectories),
            "num_frames": all_trajectories[0].shape[0],
            "num_atoms": all_trajectories[0].shape[1]
        }
    }
    
    torch.save(data_to_save, output_file)
    print(f"Successfully saved {len(all_trajectories)} trajectories to {output_file}")


if __name__ == "__main__":
    # Usage:
    preprocess_chromatin_trajectories("./ChromMovie-flow/outputs/trajectories/test01/G2M>G1", "chromatin_data.pt")
    
    

