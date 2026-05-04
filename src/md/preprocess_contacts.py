import numpy as np
import pandas as pd
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


def plot_metacells(stat_file_path, metacell_ids):
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
    plt.show()

if __name__ == "__main__":
    stat_file_path = "data/unique_cells_PATSKI_100.txt"

    cellids = get_traj_metacell_ids(stat_file_path, \
                    'SCG0088_CTGTTAAAGTCACTAA-1', 'SCG0090_CATCATAAGTAAGTGG-1', n=3, m=10)

    plot_metacells(stat_file_path, cellids)

