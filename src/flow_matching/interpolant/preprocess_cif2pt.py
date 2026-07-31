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
    return np.vstack(position_arrays)


def load_txt_coords(file_path):
    """
    Reads a space-separated text file of floating-point numbers with centers' coordinates into a 2D numpy array.
    """
    return np.loadtxt(file_path)


def preprocess_chromatin_trajectories(CM_output_path, CM_input_path, output_file, beads_per_structure=None):
    """
    CM_output_path: Path to the folder containing N subfolders (trajectories) with CM output
    CM_input_path: Path to the folder containing N subfolders (trajectories) with CM input
    output_file: Path to save the processed .pt file
    beads_per_structure: (Optional) Expected K chromatin beads to verify consistency
    """
    all_trajectories = []
    CM_output_root_path = Path(CM_output_path)
    CM_input_root_path = Path(CM_input_path)
        
    # Identify subfolders (N trajectories)
    CM_output_traj_folders = sorted([f for f in CM_output_root_path.iterdir() if f.is_dir()])
    CM_input_traj_folders = sorted([f for f in CM_input_root_path.iterdir() if f.is_dir()])
    
    print(f"Found N={len(CM_output_traj_folders)} trajectories.")
    print(f"Each trajectory has n={len([f for f in os.listdir(CM_input_traj_folders[0]) if f.endswith(".csv")])} frames.")
    example_cif = [f for f in os.listdir(os.path.join(CM_output_traj_folders[0], 'frames_cif')) if f.endswith('.cif')][0]
    example_cif_shape = read_cif(os.path.join(CM_output_traj_folders[0], 'frames_cif', example_cif)).shape
    print(f"Each frame is of shape {example_cif_shape}.")
    print(f"This amounts to K={example_cif_shape[0]} beads and K*3 = {example_cif_shape[0]*example_cif_shape[1]} values in total (+2 values for center coordinates).\n")

    for folder_CM_out, folder_CM_in in tqdm(zip(CM_output_traj_folders, CM_input_traj_folders), desc="Processing Trajectories"):
        # Identify all .cif files in chronological/ordered order (n structures)
        cif_files = sorted([f for f in os.listdir(os.path.join(folder_CM_out, 'frames_cif')) if f.endswith('.cif')])
        max_step = max([int(f.split('_')[0][4:]) for f in cif_files])
        cif_files = [f for f in cif_files if int(f.split('_')[0][4:]) == max_step] # only preprocesses the last step
        cif_files = [os.path.join(folder_CM_out, 'frames_cif', f) for f in cif_files]
        centers_file = [f for f in os.listdir(folder_CM_in) if f.endswith('.txt')]
        if len(centers_file) > 1:
            raise(Exception(f"Found more than one txt file in the folder {folder_CM_in}"))
        elif len(centers_file) == 0:
            raise(Exception(f"Found no txt file with centers coordinates in {folder_CM_in}"))
        else:
            centers_file = centers_file[0]
        centers_coords = load_txt_coords(os.path.join(folder_CM_in, centers_file))
        
        trajectory_coords = []
        
        for cif_file, center_coords in zip(cif_files, centers_coords):
            try:
                # Extract (K, 3) coordinates from cif_file, flatten them and merge with center 2D coordinates.
                # Results coords variable is of (3*K+2,) shape.
                coords = np.hstack([read_cif(cif_file).flatten(), center_coords])
                
                if beads_per_structure and coords.shape[0] != beads_per_structure:
                    print(f"Warning: {cif_file} has {coords.shape[0]} atoms, expected {beads_per_structure}")
                
                trajectory_coords.append(coords)
            except Exception as e:
                print(f"Error reading {cif_file}: {e}")

        # Convert list of arrays to a single tensor (n, 3*K+2)
        if trajectory_coords:
            traj_tensor = torch.tensor(np.array(trajectory_coords), dtype=torch.float32)
            all_trajectories.append(traj_tensor)

    # Save as a dictionary for easy loading
    data_to_save = {
        "trajectories": all_trajectories, # List of size N of (n, 3*K+2) tensors
        "metadata": {
            "num_trajectories": len(all_trajectories),
            "num_frames": all_trajectories[0].shape[0],
            "num_values": all_trajectories[0].shape[1],
            "num_beads": int((all_trajectories[0].shape[1]-2)/3)
        }
    }
    print(f"Final Torch file is of size {len(all_trajectories)} and shape {tuple(all_trajectories[0].shape)} each.")

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    torch.save(data_to_save, output_file)
    print(f"Successfully saved {len(all_trajectories)} trajectories to {output_file}")


if __name__ == "__main__":
    # Usage:
    test_name = "test02"
    phase_trans = "G2M_to_G1"
    preprocess_chromatin_trajectories(f"../../../data/md/{test_name}/CM_output/{phase_trans}", 
                                      f"../../../data/md/{test_name}/CM_input/{phase_trans}",
                                      f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt")
    
    

