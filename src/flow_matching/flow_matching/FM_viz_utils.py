import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
from torch.utils.data import DataLoader

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


def create_gif_from_figs(figs, filename, duration=200):
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


def plot_point_evolution(points_list):
    """
    Creates and returns a fig object from a list of strutures.
    """
    data = np.array(points_list)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(X, Y, color='gray', alpha=0.3, linewidth=1, zorder=1)
    
    ax.scatter(X[0, :], Y[0, :], 
               color='blue', marker='o', s=30, label='Start', zorder=3)
    
    ax.scatter(X[-1, :], Y[-1, :], 
               color='yellow', marker='o', s=50, label='End', zorder=3)
    
    ax.set_title('Evolution of 100 Points Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    return fig


def create_FM_gif(model_path, data_path, out_path):
    """
    Main function for creating a gif visualization file for the FM iterated trajectories.
    """
    device = "cpu"
    FM_model = MLP(layers=5, channels_data=7976, channels=512)
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
                                            num_steps=50, save_every=2, method="rk4")
    structures = [struct[:, -2:] for struct in structures]

    figs = []
    for i in range(1, len(structures)):
        figs.append(plot_point_evolution(structures[:i]))
    create_gif_from_figs(figs, out_path)


if __name__ == "__main__":
    test_name = "test02"
    phase_trans = "G2M_to_G1"

    viz_path = f"../../../results/FM/{test_name}/losses/"
    out_path = f"../../../results/FM/{test_name}/viz/losses.png"
    
    plot_model_losses(os.path.join(viz_path, f"{test_name}_loss_train.txt"), 
                      os.path.join(viz_path, f"{test_name}_loss_test.txt"),
                      out_path, fig_show=False)

    FM_model_path = f"../../../models/FM/{test_name}_FM_model.pt"
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    gif_path = f"../../../results/FM/{test_name}/viz/traj_FM_2D.gif"

    create_FM_gif(FM_model_path, data_path, gif_path)

