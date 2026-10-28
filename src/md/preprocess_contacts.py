import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull, distance_matrix
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def get_diameter(points):
    """
    Calculates the diameter of the convex hull around a group of points (maximal distance between points).
    """
    points = np.asarray(points)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    dist_mat = distance_matrix(hull_points, hull_points)
    return np.max(dist_mat)


def load_stat_file(stat_file_path):
    """
    Loads the file with the statistics about cells.
    First row should be reserved for column names.
    The file should contain a 'cellid', 'Phase', 'PC_1', 'PC_2' columns.
    Additional columns are also possible.
    """
    df = pd.read_csv(stat_file_path)
    df.index = range(df.shape[0])

    for col in ['cellid', 'Phase', 'PC_1', 'PC_2']:
        if col not in df.columns:
            raise(Exception(f"The statistics file does not contain '{col}' column!"))

    return df


def load_contact_data(contact_data_path):
    """
    Loads the file with the contact data.
    The input data is expected to be in bedpe format with an additional column specifying the cellid.
    """
    df = pd.read_csv(contact_data_path, header=None)
    df.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "cellid"]
    return df


def get_traj_metacell_ids_linear(stat_file_path, cellid_1, cellid_2, n=2, m=10):
    """
    Prepares a list of lists of cellids for a given trajectory between cellid_1 and cellid_2.
    The resulting list will be of length n, and each set of cells will be of size m.
    """
    df = load_stat_file(stat_file_path)

    cellid_list = list(df['cellid'])
    for cellid in [cellid_1, cellid_2]:
        if cellid not in cellid_list:
            raise(Exception(f"Given cellid ({cellid}) cannot be found in the statistics file!"))
    
    start_id = cellid_list.index(cellid_1)
    start_pos = np.array([df.loc[start_id, 'PC_1'], df.loc[start_id, 'PC_2']])
    end_id = cellid_list.index(cellid_2)
    end_pos = np.array([df.loc[end_id, 'PC_1'], df.loc[end_id, 'PC_2']])

    centers = [ start_pos*t + end_pos*(1-t) for t in np.linspace(0, 1, n)]
    metacell_ids = []
    for i in range(n):
        distances = [np.linalg.norm(centers[i] - np.array(df.loc[j, ['PC_1', 'PC_2']])) for j in range(df.shape[0])]
        cellids = [x for _, x in sorted(zip(distances, cellid_list))][:m]
        metacell_ids.append(cellids)
    
    return metacell_ids


def get_traj_metacell_ids_cubic(stat_file_path, cellid_1, cellid_2, cellid_3, n=2, m=10):
    """
    Prepares a 3 lists of lists of cellids for a trajectories between cellid_1, cellid_2 and cellid_3.
    The resulting lists will be of length n, and each set of cells will be of size m.
    The function additionally returns the 3 lists of (x,y) coordinates of the frame centers.
    """
    df = load_stat_file(stat_file_path)

    cellid_list = list(df['cellid'])
    for cellid in [cellid_1, cellid_2, cellid_3]:
        if cellid not in cellid_list:
            raise(Exception(f"Given cellid ({cellid}) cannot be found in the statistics file!"))

    anchor_cellids = [cellid_list.index(cellid) for cellid in [cellid_1, cellid_2, cellid_3]]
    anchor_positions = [np.array([df.loc[id1, 'PC_1'], df.loc[id1, 'PC_2']]) for id1 in anchor_cellids]

    t = np.array([0, 1, 2, 3])
    csX = CubicSpline(t, [pos[0] for pos in anchor_positions] + [anchor_positions[0][0]], bc_type='periodic')
    csY = CubicSpline(t, [pos[1] for pos in anchor_positions] + [anchor_positions[0][1]], bc_type='periodic')

    centers12 = [ [csX(t), csY(t)] for t in np.linspace(0, 1, n)]
    centers23 = [ [csX(t), csY(t)] for t in np.linspace(1, 2, n)]
    centers31 = [ [csX(t), csY(t)] for t in np.linspace(2, 3, n)]
    
    metacell_ids12 = []
    metacell_ids23 = []
    metacell_ids31 = []
    
    for i in range(n):
        distances = [np.linalg.norm(centers12[i] - np.array(df.loc[j, ['PC_1', 'PC_2']])) for j in range(df.shape[0])]
        cellids = [x for _, x in sorted(zip(distances, cellid_list))][:m]
        metacell_ids12.append(cellids)

        distances = [np.linalg.norm(centers23[i] - np.array(df.loc[j, ['PC_1', 'PC_2']])) for j in range(df.shape[0])]
        cellids = [x for _, x in sorted(zip(distances, cellid_list))][:m]
        metacell_ids23.append(cellids)
        
        distances = [np.linalg.norm(centers31[i] - np.array(df.loc[j, ['PC_1', 'PC_2']])) for j in range(df.shape[0])]
        cellids = [x for _, x in sorted(zip(distances, cellid_list))][:m]
        metacell_ids31.append(cellids)
        
    return metacell_ids12, metacell_ids23, metacell_ids31, centers12, centers23, centers31


def get_traj_aspect_ratio(points: list) -> bool:
    """
    Calculates SVD based aspect ratio of semi-axes on 2D point set forming an ellipse/shape.
    
    Parameters:
        points: List of 2D coordinates.
        
    Returns:
        float: Aspet ratio (1.0 - circle, 0.0 - straight line).
    """
    pts = np.array(points, dtype=float).reshape(-1, 2)
    centered = pts - np.mean(pts, axis=0)
    _, s, _ = np.linalg.svd(centered) # SVD
    
    if s[0] == 0:
        return 0.0  # All points are likely identical

    aspect_ratio = s[1] / s[0]
    return aspect_ratio


def plot_metacells(stat_file_path, metacell_ids, save_path=None):
    """
    Visualisation function for ready metacell_ids form get_traj_metacell_ids() function.
    """
    df = load_stat_file(stat_file_path)
    df['traj_metacell'] = 'Rest'

    for i, group_ids in enumerate(metacell_ids):
        df.loc[df['cellid'].isin(group_ids), 'traj_metacell'] = f'Group {i+1}'

    unique_groups = [g for g in df['traj_metacell'].unique() if g != 'Rest']
    colors = sns.color_palette("husl", len(unique_groups))
    palette = {group: color for group, color in zip(unique_groups, colors)}
    palette['Rest'] = "#6E6D6D"

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df.sort_values(by='traj_metacell', ascending=False), 
        x='PC_1', 
        y='PC_2', 
        hue='traj_metacell', 
        palette=palette,
        s=50,
        alpha=0.8,
        edgecolor='w'
    )

    plt.title('2D PCA projection with grouped metacells')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def prepare_CM_input_trajectory(metacells_ids, output_path, contact_file_path, traj_order=5):
    """
    Prepares the input for a single trajectory in a proper csv format suitable for ChromMovie simulations.
    Contact data is specified through 'contact_file_path' parameter. 
    The input data is expected to be in bedpe format with an additional column specifying the cellid.
    The output csv files are saved into the 'output_path'.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_contact = load_contact_data(contact_file_path)
    
    for i in range(len(metacells_ids)):
        df_metacell = df_contact[df_contact['cellid'].isin(metacells_ids[i])]
        df_metacell = df_metacell.drop('cellid', axis=1)
        df_metacell.to_csv(os.path.join(output_path, f'metacell_{str(i).zfill(traj_order)}.csv'), index=False)


def create_cycle_folder(output_path, n, traj_order):
    """
    Merges the existing phase trajectory folders into whole cycle trajecotries in a separate "cycle" folder.
    """
    phase_folders = [os.path.join(output_path, transition) for transition in ['G1_to_S', 'S_to_G2M', 'G2M_to_G1']]
    traj_folders = os.listdir(phase_folders[0])

    cycle_path = os.path.join(output_path, "cycle")
    if not os.path.exists(cycle_path):
        os.makedirs(cycle_path)

    metacell_files = [f for f in os.listdir(os.path.join(phase_folders[0], traj_folders[0])) if f.startswith("metacell_")]
    metacell_files.sort()
    
    for traj_folder in traj_folders:
        new_traj_folder = os.path.join(cycle_path, traj_folder)
        if not os.path.exists(new_traj_folder):
            os.makedirs(new_traj_folder)
        new_centers = []
        k = 0
        for phase_folder in phase_folders:
            for i in range(n-1):
                shutil.copyfile(os.path.join(phase_folder, traj_folder, metacell_files[i]), os.path.join(new_traj_folder, f"metacell_{str(k).zfill(traj_order)}.csv"))
                k += 1
            centers = np.loadtxt(os.path.join(phase_folder, traj_folder, f"{traj_folder}_centers_coords.txt"))
            new_centers.append(centers[:-1,:])
        new_centers = np.concatenate(new_centers)
        np.savetxt(os.path.join(new_traj_folder, f"{traj_folder}_centers_coords.txt"), new_centers)


def prepare_CM_input(output_path, contact_file_path, stat_file_path, n_traj, n=2, m=10, traj_order=5, min_traj_ratio=0.1):
    """
    Prepares the input for n_traj trajectories for each phase transition 
    in a proper csv format suitable for ChromMovie simulations.
    Contact data is specified through 'contact_file_path' parameter. 
    The input data is expected to be in bedpe format with an additional column specifying the cellid.
    The output csv files are saved into the 'output_path'.
    traj_order denotes the expected highest number of trajectories in the naming convention. traj_order=3 - "trajectory000".
    """
    df = load_stat_file(stat_file_path)
    ids_G1 = df[df['Phase'] == "G1"]['cellid']
    ids_S= df[df['Phase'] == "S"]['cellid']
    ids_G2 = df[df['Phase'] == "G2M"]['cellid']

    print("Preparing trajectory input for ChromMovie simulations...")
    for i in tqdm(range(n_traj)):
        
        ar = 0.0
        ar_counter = 0
        while ar < min_traj_ratio:
            cellid_1 = np.random.choice(ids_G1)
            cellid_2 = np.random.choice(ids_S)
            cellid_3 = np.random.choice(ids_G2)
            metacells_ids12, metacells_ids23, metacells_ids31, centers12, centers23, centers31 = \
                get_traj_metacell_ids_cubic(stat_file_path, cellid_1, cellid_2, cellid_3, n, m)
            temp = centers12[:-1] + centers23[:-1] + centers31[:-1]
            ar = get_traj_aspect_ratio(temp)
            if ar_counter > 1000:
                raise Exception(f"Could not find trajectory with aspect ratio larger than min_traj_ratio={min_traj_ratio}!")
            ar_counter += 1

        # G1 -> S
        prepare_CM_input_trajectory(metacells_ids12, \
                                    os.path.join(output_path, 'G1_to_S', f'trajectory_{str(i).zfill(traj_order)}'), \
                                    contact_file_path, traj_order=traj_order)
        np.savetxt(os.path.join(output_path, 'G1_to_S', f'trajectory_{str(i).zfill(traj_order)}', \
                                    f'trajectory_{str(i).zfill(traj_order)}_centers_coords.txt'), centers12)
        # S -> G2M
        prepare_CM_input_trajectory(metacells_ids23, \
                                    os.path.join(output_path, 'S_to_G2M', f'trajectory_{str(i).zfill(traj_order)}'), \
                                    contact_file_path, traj_order=traj_order)
        np.savetxt(os.path.join(output_path, 'S_to_G2M', f'trajectory_{str(i).zfill(traj_order)}', \
                                    f'trajectory_{str(i).zfill(traj_order)}_centers_coords.txt'), centers23)
        # G2M -> G1
        prepare_CM_input_trajectory(metacells_ids31, \
                                    os.path.join(output_path, 'G2M_to_G1', f'trajectory_{str(i).zfill(traj_order)}'), \
                                    contact_file_path, traj_order=traj_order)
        np.savetxt(os.path.join(output_path, 'G2M_to_G1', f'trajectory_{str(i).zfill(traj_order)}', \
                                    f'trajectory_{str(i).zfill(traj_order)}_centers_coords.txt'), centers31)


def clean_directory(target_dir, traj_order=5):
    """
    Deletes all the files in target_dir from the ChromMovie simulation except for the cif files of the structures from the last step.
    All of the files in the target_dir are expected to contain a specific ChromMovie simulation output.
    """
    # Safety Check: Convert to absolute path and verify existence
    target_dir = os.path.abspath(target_dir)
    if not os.path.exists(target_dir):
        print(f"Error: The directory '{target_dir}' does not exist.")
        return

    # Iterate through everything inside the target directory
    for item in os.listdir(target_dir):
        if not item.endswith("yaml"):
            item_path = os.path.join(target_dir, item)

            # Check if this is the frames_cif folder and delete all files except last step
            if item == "frames_cif":
                frames = os.listdir(item_path)
                max_step = np.max([int(x.split("_")[0][4:]) for x in frames])
                for frame in frames:
                    if not frame.startswith(f"step{str(max_step).zfill(traj_order)}"):
                        os.remove(os.path.join(item_path, frame))
                continue

            try:
                # If it's a directory (and not our protected one), delete it and its contents
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                # If it's a file or a symlink, delete it
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"[ERROR] Could not delete {item}: {e}")
    # print(f"Folder {target_dir} cleared successfully")


def clean_all_directories(target_dir, traj_order=5):
    """
    Deletes all the files from the ChromMovie simulation except for the cif files of the structures from the last step and the yaml configuration file.
    This script does it for all of the subdirectories in target_dir folder. 
    All of those subdirectories are expected to contain a specific ChromMovie simulation output.
    """
    sub_directories = os.listdir(target_dir)
    for subdir in sub_directories:
        clean_directory(os.path.join(target_dir, subdir, traj_order=traj_order))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CM input files for a test run."
    )

    # Default test name used to dynamically build the output directory path
    default_test = "test02"

    parser.add_argument(
        "-t",
        "--test-name",
        type=str,
        default=default_test,
        help=f"Name of the test run (default: {default_test})",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        default=f"../../data/md/{default_test}/CM_input/",
        help="Path to output directory/file",
    )
    parser.add_argument(
        "-s",
        "--stat-file-path",
        type=str,
        default="../../data/unique_cells_PATSKI_1k.txt",
        help="Path to unique cells stat file",
    )
    parser.add_argument(
        "-c",
        "--contact-file",
        type=str,
        default="../../data/final_dataset_downsampled_1k/filtered_ChIA-PET.bedpe",
        help="Path to contact file (.bedpe)",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=10,
        help="Number of trajectories (default: 10)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="Number of frames in each trajectory (default: 3)",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=10,
        help="Number of cells in each frame (default: 10)",
    )
    parser.add_argument(
        "-ar",
        "--min_ratio",
        type=float,
        default=0.1,
        help="Minimum aspect ratio tolerable for the trajectory (default: 0.1)",
    )

    args = parser.parse_args()

    # Automatically update default output path if a custom test_name is passed
    if (
        args.test_name != default_test
        and args.out_file == f"../../data/md/{default_test}/CM_input/"
    ):
        args.out_file = f"../../data/md/{args.test_name}/CM_input/"

    traj_order = int(np.ceil(np.log10(max(args.n_traj, args.n))))+1

    prepare_CM_input(
        output_path=args.out_file,
        contact_file_path=args.contact_file,
        stat_file_path=args.stat_file_path,
        n_traj=args.n_traj,
        n=args.n,
        m=args.m,
        traj_order=traj_order,
        min_traj_ratio=args.min_ratio
    )

    create_cycle_folder(args.out_file, args.n, traj_order=traj_order)
    
    # Save metadata to a JSON file in the output directory
    run_metadata = {"command": [sys.executable] + sys.argv, "parameters": vars(args)}
    config_path = os.path.join(args.out_file, f"{args.test_name}_run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f"Metadata saved in: {config_path}")


if __name__ == "__main__":
    main()
    

