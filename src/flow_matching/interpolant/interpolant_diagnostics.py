import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.flow_matching.interpolant.interpolant_DataLoader import TrajectoryDataset
from src.flow_matching.interpolant.interpolant import PhiTransformer

from src.md.visualization_utils import visualize_whole_genome, create_gif, get_chrom_sizes

# ---------------------------------------------------------
# 1. Prediction Helper Function
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_trajectory(model, x0, x1, num_time_steps=100, device="cpu"):
    """
    Evaluates model across dense time grid t in [0, 1] for single trajectory endpoints x0, x1.
    """
    model.eval()
    t_steps = torch.linspace(0, 1, num_time_steps, device=device).unsqueeze(1) # (T, 1)
    
    # Repeat x0 and x1 along time step dimension: (T, K*3)
    x0_exp = x0.repeat(num_time_steps, 1)
    x1_exp = x1.repeat(num_time_steps, 1)
    
    # Forward pass
    phi_pred = model(x0_exp, x1_exp, t_steps.squeeze(1)) # (T, K*3)
    mu_pred = (1 - t_steps) * x0_exp + t_steps * x1_exp + t_steps * (1 - t_steps) * phi_pred
    
    return t_steps.cpu().numpy(), mu_pred.cpu().numpy(), phi_pred.cpu().numpy()


@torch.no_grad()
def predict_mu(model, x0, x1, t):
    """
    x0: Tensor of shape (Batch, K*3)
    x1: Tensor of shape (Batch, K*3)
    t:  Scalar or Tensor of shape (Batch, 1) with values in [0, 1]
    """
    model.eval()
    
    # Format t as a tensor matching batch size
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float32, device=x0.device)
    if t.ndim == 1:
        t = t.unsqueeze(1) # (Batch, 1)

    # Predict Phi term
    phi = model(x0, x1, t.squeeze(1))
    
    # Construct mu(t) = (1-t)*x0 + t*x1 + t*(1-t)*Phi
    mu = (1 - t) * x0 + t * x1 + t * (1 - t) * phi
    return mu

# ---------------------------------------------------------
# 2. Main Plotting Routine
# ---------------------------------------------------------
def run_diagnostics_plots(model_path, data_path, loss_path, output_path, trajectory_id=0, plot_show=False):
    """
    Wrapper function that starts all of the ploting functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = PhiTransformer(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    data = torch.load(data_path)
    
    # get the first input:
    trajectories = data["trajectories"]
    traj = trajectories[trajectory_id].to(device)
    N, D = traj.shape

    x0 = traj[0:1, :]   # First frame (1, K*3+2)
    x1 = traj[-1:, :]  # Last frame (1, K*3+2)

    # Predict dense trajectory
    t_grid, mu_pred, phi_pred = evaluate_trajectory(model, x0, x1, num_time_steps=20, device=device)

    # Preparing diagnostics
    print("Plotting losses plot...")
    plot_model_losses(loss_path, loss_path, output_path, fig_show=plot_show)

    print("Ploting diagnostics plot...")
    plot_diagnostics(model_path, data_path, output_path, trajectory_id=trajectory_id, plot_show=plot_show)

    print("Plotting consecutive bead distance distribution plots...")
    plot_consecutive_bead_distr(mu_pred, output_path, plot_show=plot_show)

    print("Plotting consecutive bead angle distribution plots...")
    plot_consecutive_angle_distr(mu_pred, output_path, plot_show=plot_show)

    print("Plotting RMSD plots...")
    plot_rmsd(mu_pred, traj, output_path, plot_show=plot_show)

    print("Plotting 2D embedding plot...")
    plot_2D_map(mu_pred, traj, output_path, plot_show=plot_show)

    print("Ploting diagnostics gif...")
    # plot_diagnostics_gif(mu_pred, output_path)

    print("Diagnostic plots completed.")


def plot_diagnostics(model_path, data_path, output_path, num_samples=5, time_resolution=100, trajectory_id=0, plot_show=False):
    """
    Creates a diagnostic plot for the traned model. The plot includes the comparison of the trajectory of an example bead,
    interpolant phi magnitude (Rg) as well as the distribution of errors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = PhiTransformer(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    dataset = TrajectoryDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_t_mses = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PhiTransformer Diagnostic Dashboard", fontsize=16, fontweight='bold')

    # Iterate over test trajectories
    for idx, batch_traj in enumerate(dataloader):
        if idx >= num_samples:
            break

        traj = batch_traj[0].to(device) # Shape: (N, K*3)
        N, D = traj.shape

        x0 = traj[0:1, :]   # First frame (1, K*3)
        x1 = traj[-1:, :]  # Last frame (1, K*3)

        # Predict dense trajectory
        t_grid, mu_pred, phi_pred = evaluate_trajectory(model, x0, x1, num_time_steps=time_resolution, device=device)

        # Ground truth matching via linear interpolation index
        gt_indices = np.round(t_grid.squeeze() * (N - 1)).astype(int)
        gt_traj = traj[gt_indices].cpu().numpy() # (T, K*3)

        # Metric 1: Frame MSE vs Time step
        frame_mse = np.mean((mu_pred - gt_traj) ** 2, axis=1) # (T,)
        all_t_mses.append(frame_mse)

        # Plot 1: Coordinate Trajectories (Atom 0, X/Y/Z coords) for 1 sample
        if idx == trajectory_id:
            axes[0, 0].set_title("Ground Truth vs Predicted (Sample 0, Atom 0)")
            for coord_idx, label in enumerate(['X', 'Y', 'Z']):
                axes[0, 0].plot(t_grid, gt_traj[:, coord_idx], '--', label=f'GT {label}')
                axes[0, 0].plot(t_grid, mu_pred[:, coord_idx], '-', label=f'Pred {label}')
            axes[0, 0].set_xlabel("Time (t)")
            axes[0, 0].set_ylabel("Coordinate Value")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Magnitude of Phi term over time
            phi_magnitude = np.linalg.norm(phi_pred, axis=1)
            axes[0, 1].plot(t_grid, phi_magnitude, color='purple', label=r'$||\Phi(x_0, x_1, t)||_2$')
            axes[0, 1].set_title(r"Phi Correction Magnitude $||\Phi||$ vs $t$")
            axes[0, 1].set_xlabel("Time (t)")
            axes[0, 1].set_ylabel("L2 Norm")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

    # Plot 3: Average MSE vs Time across samples
    avg_mse_t = np.mean(all_t_mses, axis=0)
    std_mse_t = np.std(all_t_mses, axis=0)
    
    axes[1, 0].plot(t_grid, avg_mse_t, color='crimson', label="Mean MSE")
    axes[1, 0].fill_between(t_grid.squeeze(), avg_mse_t - std_mse_t, avg_mse_t + std_mse_t, color='crimson', alpha=0.2)
    axes[1, 0].set_title("Mean Squared Error (MSE) vs Continuous Time t")
    axes[1, 0].set_xlabel("Time (t)")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Error Residuals Histogram
    residuals = (np.array(all_t_mses)).flatten()
    axes[1, 1].hist(residuals, bins=30, color='teal', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Residual Error Distribution")
    axes[1, 1].set_xlabel("Frame Error")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_diagnostics.png"), dpi=300)
    if plot_show: plt.show()
    print("Diagnostics plot saved as 'model_diagnostics.png'")


def plot_diagnostics_gif(mu_pred, output_path):
    """
    Creates a trajectory gif based on a trajectory_id.
    """
    num_time_steps = mu_pred.shape[0]
    figs = []
    for i in range(num_time_steps):
        points = mu_pred[i,:-2].reshape(-1, 3) # of shape (K, 3)
        fig = visualize_whole_genome(points, resolution=2000_000, diploid=True, chrom_sizes_path="../../../chrom_sizes/mm10.txt", 
                                     p=40, title=f"Timepoint {i}/{num_time_steps}", fig_show=False, axis_range=None)
        figs.append(fig)

    gif_path = os.path.join(output_path, "model_diagnostics.gif")
    create_gif(figs, gif_path, duration=100, width=600, height=600)    


def get_indices_of_nonconsecutive(resolution, diploid=True, chrom_sizes_path="../chrom_sizes/mm10.txt"):
    """
    Calculates a list of indices of beads that are the last bead in a chromosome
    (the ones whose successor is not to be counted as consecutive bead).
    """
    _, ms = get_chrom_sizes(resolution, chrom_sizes_path)

    k = 0
    nonconsec_indices = []
    k_ploid = 2 if diploid else 1
    for i in range(len(ms)):
        for _ in range(k_ploid):
            nonconsec_indices.append(k+ms[i]-1)
            k += ms[i]
        
    return nonconsec_indices


def get_indices_of_consecutive(resolution, diploid=True, chrom_sizes_path="../chrom_sizes/mm10.txt"):
    """
    Calculates a list of indices of beads that are not the last bead in a chromosome
    (the ones whose successor is to be counted as consecutive bead).
    """
    nonconsec_indices = get_indices_of_nonconsecutive(resolution, diploid, chrom_sizes_path)
    consec_indices = [i for i in range(nonconsec_indices[-1]+1) if i not in nonconsec_indices]
    
    return consec_indices


def plot_consecutive_bead_distr(mu_pred, output_path, plot_show=False, resolution=2_000_000, diploid=True, chrom_sizes_path="../../../chrom_sizes/mm10.txt"):
    """
    Creates two plots tracking the consecutive bead distribution across mu_pred.
    """
    data_list = [mu_pred[i,:-2].reshape(-1, 3) for i in range(mu_pred.shape[0])]
    distance_history = [[np.linalg.norm(arr[i] - arr[i+1]) for i in range(arr.shape[0]-1)] for arr in data_list]
    consec_dists = get_indices_of_consecutive(resolution, diploid, chrom_sizes_path)
    distance_history = [[di for i, di in enumerate(d) if i in consec_dists] for d in distance_history]

    # Stack distances into a 2D histogram matrix (Bins vs Time)
    num_bins = 50
    hist_matrix = []

    # Global min/max to ensure consistent binning across time
    min_dist = np.min(distance_history)
    max_dist = np.max(distance_history)
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    for dists in distance_history:
        counts, _ = np.histogram(dists, bins=bin_edges, density=True)
        hist_matrix.append(counts)

    hist_matrix = np.array(hist_matrix).T  # Shape: (num_bins, time_steps)

    # Plot the Heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(
        hist_matrix, 
        aspect='auto', 
        origin='lower',
        extent=[0, len(data_list), bin_edges[0], bin_edges[-1]],
        cmap='viridis'
    )

    plt.colorbar(label='Probability Density')
    plt.xlabel('Time Step / Array Index')
    plt.ylabel('Distance Between Consecutive Points')
    plt.title('Distribution of Consecutive Point Distances Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_consecutive_bead_distr1.png"), dpi=300)
    if plot_show: plt.show()

    # Calculate statistics for each time step
    times = np.arange(len(distance_history))

    means = [np.mean(d) for d in distance_history]
    medians = [np.median(d) for d in distance_history]
    p2_5 = [np.percentile(d, 2.5) for d in distance_history]
    p97_5 = [np.percentile(d, 97.5) for d in distance_history]
    mins = [np.min(d) for d in distance_history]
    maxs = [np.max(d) for d in distance_history]

    # Plot lines and shaded region intervals
    plt.figure(figsize=(10, 5))

    # Outer range: Min to Max
    plt.fill_between(
        times, mins, maxs, color="blue", alpha=0.1, label="Full Range (Min–Max)"
    )

    # Inner 95% Interval: 2.5th to 97.5th Percentile
    plt.fill_between(
        times,
        p2_5,
        p97_5,
        color="blue",
        alpha=0.25,
        label="95% Interval (2.5%–97.5%)",
    )

    # Central Trends
    plt.plot(times, medians, label="Median Distance", color="blue", linewidth=2)
    plt.plot(
        times,
        means,
        label="Mean Distance",
        color="navy",
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel("Time Step / Array Index")
    plt.ylabel("Distance Between Consecutive Points")
    plt.title("Summary Statistics of Point Distances Over Time")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_consecutive_bead_distr2.png"), dpi=300)
    if plot_show: plt.show()


def compute_triplet_angles(arr, degrees=True):
    # Form vectors pointing from vertex p_i to p_{i-1} and p_{i+1}
    u = arr[:-2] - arr[1:-1]  # Vector to previous point (K-2, 3)
    v = arr[2:] - arr[1:-1]   # Vector to next point (K-2, 3)
    
    # Compute dot products and vector norms along axis 1
    dot_products = np.einsum('ij,ij->i', u, v)
    norms_u = np.linalg.norm(u, axis=1)
    norms_v = np.linalg.norm(v, axis=1)
    
    # Cosine theta with clipping to handle floating-point inaccuracies
    cos_theta = dot_products / (norms_u * norms_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angles = np.arccos(cos_theta)
    
    return np.degrees(angles) if degrees else angles

    # Compute across your list of time-step arrays
    angle_history = [compute_triplet_angles(arr) for arr in data_list]


def plot_consecutive_angle_distr(mu_pred, output_path, plot_show=False, resolution=2_000_000, diploid=True, chrom_sizes_path="../../../chrom_sizes/mm10.txt"):
    """
    Creates two plots tracking the consecutive bead distribution across mu_pred.
    """
    data_list = [mu_pred[i,:-2].reshape(-1, 3) for i in range(mu_pred.shape[0])]
    angle_history = [compute_triplet_angles(arr) for arr in data_list]
    nonconsec_dists = get_indices_of_consecutive(resolution, diploid, chrom_sizes_path)
    angle_history = [[di for i, di in enumerate(d) if i in nonconsec_dists or i+1 in nonconsec_dists] for d in angle_history]

    # Stack distances into a 2D histogram matrix (Bins vs Time)
    num_bins = 50
    hist_matrix = []

    # Global min/max to ensure consistent binning across time
    min_dist, max_dist = 0, 180
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    for dists in angle_history:
        counts, _ = np.histogram(dists, bins=bin_edges, density=True)
        hist_matrix.append(counts)

    hist_matrix = np.array(hist_matrix).T  # Shape: (num_bins, time_steps)

    # Plot the Heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(
        hist_matrix, 
        aspect='auto', 
        origin='lower',
        extent=[0, len(data_list), bin_edges[0], bin_edges[-1]],
        cmap='viridis'
    )

    plt.colorbar(label='Probability Density')
    plt.xlabel('Time Step / Array Index')
    plt.ylabel('Angle Between Consecutive Triplets of Points')
    plt.title('Distribution of Consecutive Point Angles Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_consecutive_angle_distr1.png"), dpi=300)
    if plot_show: plt.show()

    # Calculate statistics for each time step
    times = np.arange(len(angle_history))

    means = [np.mean(d) for d in angle_history]
    medians = [np.median(d) for d in angle_history]
    p2_5 = [np.percentile(d, 2.5) for d in angle_history]
    p97_5 = [np.percentile(d, 97.5) for d in angle_history]
    mins = [np.min(d) for d in angle_history]
    maxs = [np.max(d) for d in angle_history]

    # Plot lines and shaded region intervals
    plt.figure(figsize=(10, 5))

    # Outer range: Min to Max
    plt.fill_between(
        times, mins, maxs, color="blue", alpha=0.1, label="Full Range (Min–Max)"
    )

    # Inner 95% Interval: 2.5th to 97.5th Percentile
    plt.fill_between(
        times,
        p2_5,
        p97_5,
        color="blue",
        alpha=0.25,
        label="95% Interval (2.5%–97.5%)",
    )

    # Central Trends
    plt.plot(times, medians, label="Median Distance", color="blue", linewidth=2)
    plt.plot(
        times,
        means,
        label="Mean Distance",
        color="navy",
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel("Time Step / Array Index")
    plt.ylabel("Angles Between Consecutive Triplets of Points")
    plt.title("Summary Statistics of Point Angles Over Time")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_consecutive_angle_distr2.png"), dpi=300)
    if plot_show: plt.show()


def plot_rmsd(mu_pred, orig_traj, output_path, plot_show=False):
    """
    Creates a heatmap of RMSD values between the inferred trajectory and original structures from ChromMovie simulation.
    Additionally, a heatmap of RMSD between linear trajectory and original, as well as the difference between the two is created.
    """
    list_pred = [mu_pred[i,:-2].reshape(-1, 3) for i in range(mu_pred.shape[0])]
    list_orig = [orig_traj[i,:-2].reshape(-1, 3) for i in range(orig_traj.shape[0])]
    list_linear = [list_orig[0]*(1-t)+list_orig[-1]*t for t in np.linspace(0, 1, len(list_pred))]

    # Convert lists to 3D arrays: (M, K, 3) and (N, K, 3)
    A = np.array(list_pred)
    B = np.array(list_orig)
    C = np.array(list_linear)

    # Vectorized pairwise distance calculation via broadcasting
    # Shape shift: (M, 1, K, 3) - (1, N, K, 3) -> (M, N, K, 3)
    diff = A[:, None, :, :] - B[None, :, :, :]
    squared_distances = np.sum(diff**2, axis=-1)  # Shape: (M, N, K)
    rmsd_matrix = np.sqrt(np.mean(squared_distances, axis=-1))  # Shape: (M, N)

    diff2 = C[:, None, :, :] - B[None, :, :, :]
    squared_distances2 = np.sum(diff2**2, axis=-1)  # Shape: (M, N, K)
    rmsd_matrix2 = np.sqrt(np.mean(squared_distances2, axis=-1))  # Shape: (M, N)

    # Plot Heatmap with predicted
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        rmsd_matrix,
        cmap="RdYlGn_r",
        xticklabels=1,
        yticklabels=1,
        cbar_kws={"label": "RMSD"},
    )

    plt.xlabel("Original Trajectory Structure Index")
    plt.ylabel("Model Trajectory Index")
    plt.title("Pairwise (All-vs-All) RMSD Heatmap for predicted trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_RMSD_with_predicted.png"), dpi=300)
    if plot_show: plt.show()

    # Plot Heatmap with linear
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        rmsd_matrix2,
        cmap="RdYlGn_r",
        xticklabels=1,
        yticklabels=1,
        cbar_kws={"label": "RMSD"},
    )

    plt.xlabel("Original Trajectory Structure Index")
    plt.ylabel("Model Trajectory Index")
    plt.title("Pairwise (All-vs-All) RMSD Heatmap for linear interpolation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_RMSD_with_linear.png"), dpi=300)
    if plot_show: plt.show()

    # Plot Heatmap of differences between RMSDs from linear and predicted
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        rmsd_matrix - rmsd_matrix2,
        cmap="RdYlGn_r",
        xticklabels=1,
        yticklabels=1,
        cbar_kws={"label": "RMSD difference"},
    )

    plt.xlabel("Original Trajectory Structure Index")
    plt.ylabel("Model Trajectory Index")
    plt.title("Pairwise (All-vs-All) difference between linear and predicted RMSD Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_RMSD_difference.png"), dpi=300)
    if plot_show: plt.show()


def plot_2D_map(mu_pred, orig_traj, output_path, plot_show=False):
    """
    Uses last two coordinates of the prediction to plot the 2D trajectory and compares it to the linear interpolation.
    """
    list_pred = [mu_pred[i,-2:] for i in range(mu_pred.shape[0])]
    list_orig = [orig_traj[i,-2:] for i in range(orig_traj.shape[0])]
    list_linear = [list_orig[0]*(1-t)+list_orig[-1]*t for t in np.linspace(0, 1, len(list_pred))]

    pred = np.vstack(list_pred)
    linear = np.vstack(list_linear)
    orig = np.vstack(list_orig)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Model prediction trajectory
    pred = np.array(list_pred)
    ax.plot(
        pred[:, 0],
        pred[:, 1],
        c="blue",
        marker="o",
        label="Model prediction"
    )

    # Linear interpolation trajectory
    linear = np.array(list_linear)
    ax.plot(
        linear[:, 0],
        linear[:, 1],
        c="gray",
        marker="o",
        label="Linear interpolation"
    )

    # Original/checkpoint coordinates
    orig = np.array(list_orig)
    ax.scatter(
        orig[:, 0],
        orig[:, 1],
        s=80,
        c="black",
        zorder=3,
        label="Original checkpoints"
    )

    # Label first and last checkpoints
    ax.annotate(
        "x0",
        xy=orig[0],
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=12
    )

    ax.annotate(
        "x1",
        xy=orig[-1],
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=12
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.legend()
    plt.savefig(os.path.join(output_path, "model_2D_map.png"), dpi=300)
    if plot_show: plt.show()


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
    plt.title('Interpolant Model Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Display the plot
    plt.tight_layout()
    if fig_show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_path, "train_loss.png"))

        
# Execute Script
if __name__ == "__main__":

    test_name = "test02"
    phase_trans = "G2M_to_G1"
    trajectory_id = 2
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    model_path = f"../../../models/interpolant/{test_name}_interp_model.pt"
    train_loss_path = f"../../../results/interpolant/{test_name}/losses/{test_name}_loss_train.txt"
    output_path = f"../../../results/interpolant/{test_name}/traj{str(trajectory_id).zfill(3)}"
    
    run_diagnostics_plots(model_path, data_path, train_loss_path, output_path, trajectory_id=trajectory_id, plot_show=False)

    
