import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
import argparse

from src.flow_matching.flow_matching.FM_model import MLP
from src.flow_matching.interpolant.interpolant_DataLoader import TrajectoryDataset
from src.flow_matching.flow_matching.solvers import iterate_trajectory_pytorch


def plot_model_losses(train_file, test_file, out_path, fig_show=True):
    """
    Plots model train and test losses on a single plot.
    """
    # Read the train losses into a list of floats
    with open(train_file, 'r') as f:
        train_losses = [float(line.strip()) for line in f if line.strip()]
        
    # Read the test losses into a list of floats
    with open(test_file, 'r') as f:
        test_losses = [float(line.strip()) for line in f if line.strip()]
        
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2, linestyle='--')
    
    # Customize the plot with labels, title, and grid
    plt.title('FM Model Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Display the plot
    plt.tight_layout()
    if fig_show:
        plt.show()
    else:
        plt.savefig(out_path)
        print(f"Succcessfully saved losses png at {out_path}")


def create_gif_from_figs(figs, filename, duration=300):
    """
    Converts a list of matplotlib Figure objects into a GIF.
    
    Parameters:
    - figs: List of matplotlib.figure.Figure objects.
    - filename: Output file name for the GIF.
    - duration: Duration of each frame in milliseconds.
    """
    images = []
    
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0) 
        image = Image.open(buf).convert('RGBA')
        images.append(image)
        plt.close(fig)
       
    if images:
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Successfully saved {len(figs)} frames to {filename}")


def plot_point_evolution(points_list, xlim, ylim):
    """
    Creates and returns a fig object from a list of structures,
    including a background KDE density map for the final positions.
    """
    data = np.array(points_list)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    x_final = X[-1, :]
    y_final = Y[-1, :]
    
    # Create a 100x100 grid over the specified xlim and ylim bounds
    xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    final_points = np.vstack([x_final, y_final])
    
    # Fit the Gaussian KDE and evaluate it on the grid
    kernel = gaussian_kde(final_points)
    density = np.reshape(kernel(positions).T, xx.shape)
    
    _ = ax.pcolormesh(
        xx, yy, density, 
        cmap='viridis', 
        alpha=0.6, 
        shading='gouraud',
        zorder=0
    )

    ax.plot(X, Y, color='black', alpha=0.2, linewidth=2, zorder=1)
    
    ax.scatter(X[0, :], Y[0, :], 
               color='blue', edgecolor='black', marker='o', s=30, label='Start', zorder=3)
    
    ax.scatter(X[-1, :], Y[-1, :], 
               color='yellow', edgecolor='black', marker='o', s=50, label='End', zorder=3)
    
    ax.set_title(f'Evolution of {data.shape[1]} FM Trajectories Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
        
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0.5)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    
    return fig


def plot_2d_velocity_field(model_path, data_path, t, out_path, grid_size=20, sample_idx=0, fig_show=False):
    """
    Plots a 2D velocity field for the last two dimensions of a dataset.
    
    Args:
        model: PyTorch model returning velocity field (expects x of shape (N, D) and time t)
        data_x0: Tensor of shape (B, D) representing dataset samples
        t: Scalar or tensor representing time step
        grid_size: Resolution of the vector field grid (grid_size x grid_size arrows)
        sample_idx: Index of sample in data_x0 to use for the fixed D-2 dimensions
        device: 'cpu' or 'cuda'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FM_model = MLP(layers=5, channels_data=7976, channels=512).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    FM_model.load_state_dict(checkpoint["model_state_dict"])
    FM_model.eval()

    dataset = TrajectoryDataset(data_path)
    full_batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=full_batch_size, shuffle=True)
    batch_trajectories = next(iter(dataloader)).to(device)
    data_x0 = batch_trajectories[:, 0, :]
    
    data_x0 = data_x0.to(device)

    # Determine bounds for the last two dimensions (index -2 and -1)
    x_min, x_max = data_x0[:, -2].min().item(), data_x0[:, -2].max().item()
    y_min, y_max = data_x0[:, -1].min().item(), data_x0[:, -1].max().item()

    # Add 10% padding around data boundaries
    pad_x = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    pad_y = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

    # Build 2D grid coordinates
    grid_x = torch.linspace(x_min - pad_x, x_max + pad_x, grid_size, device=device)
    grid_y = torch.linspace(y_min - pad_y, y_max + pad_y, grid_size, device=device)
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="ij")

    # Flatten coordinates into a batch of 2D points: shape (grid_size * grid_size, 2)
    grid_2d = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)
    num_points = grid_2d.shape[0]

    # Construct input tensor for shape (num_points, D)
    D = data_x0.shape[1]
    if D > 2:
        # Freeze prefix dimensions using the values from sample_idx
        prefix = data_x0[sample_idx, :-2].unsqueeze(0).expand(num_points, -1)
        grid_input = torch.cat([prefix, grid_2d], dim=-1)
    else:
        grid_input = grid_2d

    # Prepare time tensor matching grid batch size
    if isinstance(t, (int, float)):
        t_tensor = torch.full((num_points, 1), fill_value=t, device=device)
    else:
        t_tensor = t.to(device).expand(num_points, -1)

    # Evaluate velocities through the model
    with torch.no_grad():
        # Adapt signature if your model takes model(t, x) instead of model(x, t)
        velocities = FM_model(grid_input, t_tensor)

    # Extract horizontal (dim -2) and vertical (dim -1) velocity components
    v_x = velocities[:, -2].reshape(grid_size, grid_size).cpu().numpy()
    v_y = velocities[:, -1].reshape(grid_size, grid_size).cpu().numpy()

    X = mesh_x.cpu().numpy()
    Y = mesh_y.cpu().numpy()

    # Calculate magnitude for coloring arrow strength
    speed = np.hypot(v_x, v_y)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Quiver plot draws the directional arrows colored by speed
    q = ax.quiver(X, Y, v_x, v_y, speed, cmap="viridis", angles="xy", scale_units="xy")
    fig.colorbar(q, ax=ax, label="Velocity Magnitude (Strength)")

    # Overlay original data points projected onto the last two dimensions
    ax.scatter(
        data_x0[:, -2].cpu().numpy(),
        data_x0[:, -1].cpu().numpy(),
        c="red",
        alpha=0.3,
        s=15,
        label="Data ($x_0$)",
    )

    ax.set_xlabel("Dimension D-2")
    ax.set_ylabel("Dimension D-1")
    ax.set_title(f"Velocity Field at t = {t}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if fig_show:
        plt.show()
    else:
        plt.savefig(out_path)
        print(f"Succcessfully saved 2D velocity field png at {out_path}")


def create_FM_gif(model_path, data_path, out_path, num_steps=10, save_every=1, method="rk4", end_time=0.1):
    """
    Main function for creating a gif visualization file for the FM iterated trajectories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FM_model = MLP(layers=5, channels_data=7976, channels=512).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    FM_model.load_state_dict(checkpoint["model_state_dict"])
    FM_model.eval()

    dataset = TrajectoryDataset(data_path)
    full_batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=full_batch_size, shuffle=True)
    batch_trajectories = next(iter(dataloader)).to(device)
    # batch_trajectories shape: (Total_Dataset_Size, N, D)
    B, N, D = batch_trajectories.shape
    data_x0 = batch_trajectories[:, 0, :]   # First frame (B, D)

    structures = iterate_trajectory_pytorch(FM_model, data_x0,
                                            num_steps=num_steps, save_every=save_every, 
                                            method=method, end_time=end_time)
    structures = [struct[:, -2:] for struct in structures]

    data = np.array(structures)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    xlim = (np.min(X), np.max(X))
    ylim = (np.min(Y), np.max(Y))
        
    figs = []
    for i in range(1, len(structures)):
        figs.append(plot_point_evolution(structures[:i], xlim, ylim))
    create_gif_from_figs(figs, out_path, duration=50)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training losses and generate Flow Matching (FM) trajectory GIF."
    )
    parser.add_argument(
        "--test-name",
        "-t",
        type=str,
        default="test01",
        help="Name of the test directory (default: 'test01')",
    )
    parser.add_argument(
        "--phase-trans",
        "-p",
        type=str,
        default="G1_to_S",
        help="Phase transition identifier (default: 'G1_to_S')",
    )
    parser.add_argument(
        "--fig-show",
        action="store_true",
        help="Display the loss plot interactively (default: False)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../../data/md",
        help="Base directory for processed data (default: '../../../data/md')",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../../../models/FM",
        help="Base directory for FM models (default: '../../../models/FM')",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../../../results/FM",
        help="Base directory for results and visualizations (default: '../../../results/FM')",
    )

    args = parser.parse_args()

    # Define loss and plot paths
    viz_path = os.path.join(args.results_dir, args.test_name, "losses")
    out_path = os.path.join(args.results_dir, args.test_name, "viz", "losses.png")

    train_loss_file = os.path.join(viz_path, f"{args.test_name}_loss_train.txt")
    test_loss_file = os.path.join(viz_path, f"{args.test_name}_loss_test.txt")

    plot_model_losses(train_loss_file, test_loss_file, out_path, fig_show=args.fig_show)

    # Define model, data, and gif paths
    FM_model_path = os.path.join(args.models_dir, f"{args.test_name}_FM_model.pt")
    data_path = os.path.join(
        args.data_dir, args.test_name, "processed", f"{args.phase_trans}_data.pt"
    )
    vel_path = os.path.join(args.results_dir, args.test_name, "viz", "velocity_field_FM_2D.png")
    gif_path = os.path.join(args.results_dir, args.test_name, "viz", "traj_FM_2D.gif")

    plot_2d_velocity_field(FM_model_path, data_path, 0, vel_path)
    create_FM_gif(FM_model_path, data_path, gif_path)


if __name__ == "__main__":
    main()

