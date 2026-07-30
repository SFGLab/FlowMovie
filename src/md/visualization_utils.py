import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.interpolate import CubicSpline, splev, splprep

import io
from PIL import Image
# Worked with:
# /home/baneckik/Programs/miniconda3/bin/python3.13 -m pip install -U "plotly==5.3.1"
# and
# /home/baneckik/Programs/miniconda3/bin/python3.13 -m pip install -U "kaleido==0.2.1"
import plotly 
import kaleido


def load_txt_coords(file_path):
    """
    Reads a space-separated text file of floating-point numbers with centers' coordinates into a 2D numpy array.
    """
    return np.loadtxt(file_path)


def get_chrom_sizes(resolution, chrom_sizes_path="../chrom_sizes/mm10.txt"):
    """
    Returns a tuple with a list of chromosome sizes in basepairs and list of chromosome sizes in beads.
    Order of chroms is as follows: chr1, chr2, ..., chr19, chrX.
    """
    chroms = ["chr"+str(i) for i in range(1, 20)] + ["chrX"]
    chrom_sizes = [int(pd.read_csv(chrom_sizes_path, header=None, index_col=0, sep="\t").loc[chrom.split("-")[0], 1]) \
                   for chrom in chroms]
    ms = [int(np.ceil(chrom_size/resolution)) for chrom_size in chrom_sizes]
    return chrom_sizes, ms


def visualize_single_chrom(points, title="", fig_show=True, axis_range=None):
    """
    Interactive plotly visualization of a single chromosome in the form of (m,3) np.array.
    Consecutive points (rows in points array) are assumed to be connected).
    axis_range is expected to be in the form of [x_min, x_max, y_min, y_max, z_min, z_max].
    """
    n = points.shape[0]
    edge_indices = np.vstack([[i, i+1] for i in range(n-1)])
    
    # Construct x, y, z coordinates for lines separated by None
    edge_x, edge_y, edge_z = [], [], []
    for start_idx, end_idx in edge_indices:
        edge_x.extend([points[start_idx, 0], points[end_idx, 0], None])
        edge_y.extend([points[start_idx, 1], points[end_idx, 1], None])
        edge_z.extend([points[start_idx, 2], points[end_idx, 2], None])
    
    # 3. Create interactive WebGL traces
    scatter_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=np.array(range(points.shape[0])), colorscale='Viridis', opacity=0.8)
    )
    
    line_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    )
    
    # Render plot
    if axis_range is None:
        axis_range_plot = [np.min(points[:,0]), np.max(points[:,0]),
                           np.min(points[:,1]), np.max(points[:,1]),
                           np.min(points[:,2]), np.max(points[:,2])]
    else:
        axis_range_plot = axis_range
    fig = go.Figure(data=[scatter_trace, line_trace])
    fig.update_layout(scene=dict(
            aspectmode='data',
            xaxis=dict(range=[axis_range_plot[0], axis_range_plot[1]]),
            yaxis=dict(range=[axis_range_plot[2], axis_range_plot[3]]),
            zaxis=dict(range=[axis_range_plot[4], axis_range_plot[5]]),
        ), 
        margin=dict(l=0, r=0, b=0, t=40), title=title, showlegend=False)

    if fig_show: fig.show()
    return fig


def visualize_whole_genome(points, resolution, diploid=False, chrom_sizes_path="../chrom_sizes/mm10.txt", p=0, title="", fig_show=True, axis_range=None):
    """
    Interactive plotly visualization of a single chromosome in the form of (m,3) np.array.
    Consecutive points (rows in points array) within particular chromosomes (based on chrom_sizes_path), are assumed to be connected).
    
    If diploid==False only a single chromosome of each pair is assumed starting from chr1 and ending on chrX.
    If diploid==True the order of chromosomes is assumed to be chr1, chr1, chr2, chr2, chr3, ..., chrX, chrX.
    Parameter p signifies the percentage (integer from 0 to 100) of the points to exclude due to them lying to close to the center. 
    By default p=0, meaning that all of the points will be shown.
    axis_range is expected to be in the form of [x_min, x_max, y_min, y_max, z_min, z_max].
    """
    _, ms = get_chrom_sizes(resolution, chrom_sizes_path)
    
    if (not diploid and np.sum(ms) != points.shape[0]) or (diploid and np.sum(ms)*2 != points.shape[0]):
        error_message = f"Number of points: ({points.shape[0]}) does not match the expected number of beads ({np.sum(ms)}) for this resolution ({resolution})."
        error_message += "\nPlease check if the resolution and ploidity are correctly set."
        raise(Exception(error_message))

    k = 0
    edge_indices = []
    k_ploid = 2 if diploid else 1
    for i in range(len(ms)):
        for _ in range(k_ploid):
            for j in range(ms[i]):
                if j != ms[i]-1:
                    edge_indices.append([k, k+1])
                k += 1
    
    edge_indices = np.vstack(edge_indices)

    # Potentially excluding p percent of point lying too close to the center
    if p==0:
        is_outside = np.array([True]*points.shape[0])
    else:
        center_of_mass = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center_of_mass, axis=1)
        threshold_distance = np.percentile(distances, p)
        is_outside = distances > threshold_distance

    # Prepare color palette
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    point_colors = []
    color = 0
    for i in range(len(ms)):
        for _ in range(k_ploid):
            for j in range(ms[i]):
                point_colors.append(palette[color])
                if j == ms[i]-1:
                    color += 1
                    color = color%len(palette)

    
    # Construct x, y, z coordinates for lines separated by None
    edge_x, edge_y, edge_z = [], [], []
    for start_idx, end_idx in edge_indices:
        if is_outside[start_idx] or is_outside[end_idx]:
            edge_x.extend([points[start_idx, 0], points[end_idx, 0], None])
            edge_y.extend([points[start_idx, 1], points[end_idx, 1], None])
            edge_z.extend([points[start_idx, 2], points[end_idx, 2], None])
    
    # Create interactive WebGL traces
    points_plot = points[is_outside,:]
    scatter_trace = go.Scatter3d(
        x=points_plot[:, 0], y=points_plot[:, 1], z=points_plot[:, 2],
        mode='markers',
        marker=dict(size=2, color=point_colors, opacity=0.8)
    )
    
    line_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=3),
        hoverinfo='none'
    )
    
    # Render plot
    if axis_range is None:
        axis_range_plot = [np.min(points[:,0]), np.max(points[:,0]),
                           np.min(points[:,1]), np.max(points[:,1]),
                           np.min(points[:,2]), np.max(points[:,2])]
    else:
        axis_range_plot = axis_range
    fig = go.Figure(data=[scatter_trace, line_trace])
    fig.update_layout(scene=dict(
            aspectmode='data',
            xaxis=dict(range=[axis_range_plot[0], axis_range_plot[1]]),
            yaxis=dict(range=[axis_range_plot[2], axis_range_plot[3]]),
            zaxis=dict(range=[axis_range_plot[4], axis_range_plot[5]]),
        ), 
        margin=dict(l=0, r=0, b=0, t=40), title=title, showlegend=False)
    
    if fig_show: fig.show()
    return fig

    
def create_gif(figs, gif_path, duration=100, width=600, height=600):
    """
    Generates a gif out of the list of figures (figs) generated by the visualize_single_chrom() or visualize_whole_genome().
    duration is per frame in milliseconds.
    width and height represent the dimensions of the image and the gif.
    """
    frames = []

    for fig in tqdm(figs):
        img_bytes = fig.to_image(format="png", width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))
        frames.append(img)
    
    frames[0].save(gif_path, save_all=True,
        append_images=frames[1:], duration=duration, loop=0
    )


def get_smooth_trajectory(pts, m2=None):
    """Fits a parametric cubic spline to 2D points (m, 2) and returns smooth (x, y) coordinates.
    :m2: number of intermediary points of the smoothed trajectory.
    """
    x, y = pts[:, 0], pts[:, 1]
    m = len(x)
    if m2 is None:
        m2 = m*3 + 1

    t = np.array(range(m))
    csX = CubicSpline(t, x)
    csY = CubicSpline(t, y)

    x_smooth = [csX(t) for t in np.linspace(0, m-1, m2)]
    y_smooth = [csY(t) for t in np.linspace(0, m-1, m2)]
        
    return x_smooth, y_smooth


def visualize_CM_input(CM_input_path, output_path):
    """
    
    """
    # Creating an output folder if does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Preparing centers coordinates as list 'centers'
    # The list is of size 3 (for each phase transition), consisting of lists of size of the number of trajectories, 
    # consisting of np.arrays of the centers of this trajectory.
    phase_folders = os.listdir(CM_input_path)
    centers_list = []
    for phase_folder in phase_folders:
        trajectories_folders = os.listdir(os.path.join(CM_input_path, phase_folder))
        centers_phase = []
        for traj_folder in trajectories_folders:
            txt_coord_file = [x for x in os.listdir(os.path.join(CM_input_path, phase_folder, traj_folder)) if x.endswith(".txt")][0]
            centers_traj = load_txt_coords(os.path.join(CM_input_path, phase_folder, traj_folder, txt_coord_file))
            centers_phase.append(centers_traj)
        centers_list.append(centers_phase)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Distinct colors for groups
    group_names = ["Group 1", "Group 2", "Group 3"]

    # --- A. Save individual plots for each group ---
    for group_idx, group, phase_trans in zip(range(len(centers_list)), centers_list, phase_folders):
        fig, ax = plt.subplots(figsize=(8, 6))

        for traj in group:
            # Get interpolated points
            x_smooth, y_smooth = get_smooth_trajectory(traj)

            # Plot smooth spline curve
            ax.plot(x_smooth, y_smooth, color=colors[group_idx], alpha=0.8, lw=2)
            # Plot original waypoint dots
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                color=colors[group_idx],
                s=15,
                alpha=0.5,
                zorder=3,
            )

        ax.set_title(
            f"Trajectories - {" ".join(phase_trans.split("_"))}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)

        filename = os.path.join(output_path, f"trajectories_{phase_trans}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")


    # --- B. Save combined plot containing all groups ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for group_idx, group in enumerate(centers_list):
        for i, traj in enumerate(group):
            x_smooth, y_smooth = get_smooth_trajectory(traj)

            # Attach label only to the first line of each group for clean legend
            label = " ".join(phase_folders[group_idx].split("_")) if i == 0 else None

            ax.plot(
                x_smooth,
                y_smooth,
                color=colors[group_idx],
                alpha=0.8,
                lw=2,
                label=label,
            )
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                color=colors[group_idx],
                s=12,
                alpha=0.4,
                zorder=3,
            )

    ax.set_title(
        "All Trajectories Combined", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Phase transitions", loc="best")

    combined_filename = os.path.join(output_path, "trajectories_all_phases.png")
    plt.savefig(combined_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_filename}")
    

if __name__ == "__main__":
    visualize_CM_input("../../data/md/test02/CM_input", "../../data/md/test02/figures")
    