import os
import numpy as np
import argparse
from openmm.app import PDBxFile
from tqdm import tqdm


def mmcif2npy(positions_path):
    """
    Extracts numpy array (n,3) from cif file. 
    Positions are in nanometers hence multiplication by 10 to convert to Angstroms.
    """
    positions = PDBxFile(positions_path).positions
    m = len(positions)
    array = np.zeros((m, 3))
    for i, p in enumerate(positions):
        array[i, :] = [p[j].real*10 for j in range(3)] 
    return array


def get_last_step(frames_cif_path):
    """
    Calculates the last step of the simulation based on the files' names and returns it as an integer.
    The filenames are expected in the format: step099_frame000.cif.
    """
    file_names = os.listdir(frames_cif_path)
    steps = [int(x.split("_")[0][4:]) for x in file_names]
    return int(np.max(steps))


def get_last_frame(frames_cif_path):
    """
    Calculates the last frame of the simulation based on the files' names and returns it as an integer.
    The filenames are expected in the format: step099_frame000.cif.
    """
    file_names = os.listdir(frames_cif_path)
    steps = [int(x.split("_frame")[-1].split(".cif")[0]) for x in file_names]
    return int(np.max(steps))


def create_npy_arrays(input_folder, output_folder):
    """
    Convert all of the cif files contained in the input folder into two numpy arrays: x0 (beginning) and x1 (end)
    Creates x0 and x1 numpy arrays and saves them in the output_folder as x0.npy and x1.npy.
    Each row of x0 stores flatten coordinates of the first frame of the simulation (in its final step).
    Each row of x1 stores flatten coordinates of the last frame of the simulation (in its final step).
    """
    traj_folders = os.listdir(input_folder)
    x0_rows = []
    x1_rows = []
    for traj_folder in tqdm(traj_folders):
        curr_folder = os.path.join(input_folder, traj_folder, "frames_cif")
        last_step = get_last_step(curr_folder)
        last_frame = get_last_frame(curr_folder)
        x0_filename = f"step{str(last_step).zfill(3)}_frame{str(last_frame).zfill(3)}.cif"
        x1_filename = f"step{str(last_step).zfill(3)}_frame000.cif"
        x0_rows.append(mmcif2npy(os.path.join(curr_folder, x0_filename)).flatten())
        x1_rows.append(mmcif2npy(os.path.join(curr_folder, x1_filename)).flatten())
    x0 = np.vstack(x0_rows)
    x1 = np.vstack(x1_rows)

    # Saving npy files to output_folder:
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    np.save(os.path.join(output_folder, "x0.npy"), x0)
    np.save(os.path.join(output_folder, "x1.npy"), x1)


if __name__=="__main__":
    """
    Convert all of the cif files contained in the input folder into two numpy arrays: x0 (beginning) and x1 (end).
    Each row of x0 stores flatten coordinates of the first frame of the simulation (in its final step).
    Each row of x1 stores flatten coordinates of the last frame of the simulation (in its final step).
    """

    # Example use:
    # python prepare_cif2npy.py -i ../../../FM_tutorial/data/test01/G1_to_S/ -o ../../data/trajectories_npy/test01/G1_to_S

    parser = argparse.ArgumentParser(
        description="Convert all of the cif files contained in the input folder into two numpy arrays: x0 (beginning) and x1 (end)."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output folder for numpy files",
    )
    args = parser.parse_args()

    create_npy_arrays(args.input, args.output)


