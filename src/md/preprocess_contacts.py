import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull, distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_diameter(points):
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


def get_traj_metacell_ids(stat_file_path, cellid_1, cellid_2, n=2, m=10):
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


def prepare_CM_input_trajectory(output_path, contact_file_path, stat_file_path, cellid_1, cellid_2, n=2, m=10):
    """
    Prepares the input for a single trajectory in a proper csv format suitable for ChromMovie simulations.
    It takes parameters of particular trajectory as input (in particular start and end cell - cellid_1 and cellid_2).
    Contact data is specified through 'contact_file_path' parameter. 
    The input data is expected to be in bedpe format with an additional column specifying the cellid.
    The output csv files are saved into the 'output_path'.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metacells_ids = get_traj_metacell_ids(stat_file_path, cellid_1, cellid_2, n, m)
    df_contact = load_contact_data(contact_file_path)
    
    # print("Preparing csv contact files for the metacells...")
    for i in range(n):
        df_metacell = df_contact[df_contact['cellid'].isin(metacells_ids[i])]
        df_metacell = df_metacell.drop('cellid', axis=1)
        df_metacell.to_csv(os.path.join(output_path, f'metacell_{str(i).zfill(3)}.csv'), index=False)


def prepare_CM_input(output_path, contact_file_path, stat_file_path, n_traj, n=2, m=10):
    """
    Prepares the input for n_traj trajectories for each phase transition 
    in a proper csv format suitable for ChromMovie simulations.
    Contact data is specified through 'contact_file_path' parameter. 
    The input data is expected to be in bedpe format with an additional column specifying the cellid.
    The output csv files are saved into the 'output_path'.
    """
    df = load_stat_file(stat_file_path)
    ids_G1 = df[df['Phase'] == "G1"]['cellid']
    ids_S= df[df['Phase'] == "S"]['cellid']
    ids_G2 = df[df['Phase'] == "G2M"]['cellid']

    print("Preparing trajectory input for ChromMovie simulations...")
    for i in tqdm(range(n_traj)):
        # G1 -> S
        cellid_1 = np.random.choice(ids_G1)
        cellid_2 = np.random.choice(ids_S)
        prepare_CM_input_trajectory(os.path.join(output_path, 'G1>S', f'trajectory_{str(i).zfill(3)}'), \
                                    contact_file_path, stat_file_path,\
                                    cellid_1, cellid_2, n=n, m=m)
        # S -> G2M
        cellid_1 = np.random.choice(ids_S)
        cellid_2 = np.random.choice(ids_G2)
        prepare_CM_input_trajectory(os.path.join(output_path, 'S>G2M', f'trajectory_{str(i).zfill(3)}'), \
                                    contact_file_path, stat_file_path,\
                                    cellid_1, cellid_2, n=n, m=m)
        # G2M -> G1
        cellid_1 = np.random.choice(ids_G2)
        cellid_2 = np.random.choice(ids_G1)
        prepare_CM_input_trajectory(os.path.join(output_path, 'G2M>G1', f'trajectory_{str(i).zfill(3)}'), \
                                    contact_file_path, stat_file_path,\
                                    cellid_1, cellid_2, n=n, m=m)
        
        
if __name__ == "__main__":
    out_file = "./ChromMovie-flow/outputs/md/test00"
    stat_file_path = "./data/unique_cells_PATSKI_100.txt"
    contact_file = "data/final_dataset_downsampled_100/filtered_ChIA-PET.bedpe"
    cellid1, cellid2 = 'SCG0088_CTGTTAAAGTCACTAA-1', 'SCG0090_CATCATAAGTAAGTGG-1'

    # cellids = get_traj_metacell_ids(stat_file_path, \
    #                 'SCG0088_CTGTTAAAGTCACTAA-1', 'SCG0090_CATCATAAGTAAGTGG-1', n=3, m=10)
    # plot_metacells(stat_file_path, cellids, save_path="./test00.png")

    prepare_CM_input(out_file, contact_file, stat_file_path, n_traj=10, n=3, m=10)
