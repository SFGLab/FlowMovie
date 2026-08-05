import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from interpolant_DataLoader import TrajectoryDataset
from interpolant import PhiTransformer


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

# ---------------------------------------------------------
# 2. Main Plotting Routine
# ---------------------------------------------------------
def run_diagnostics(model_path, data_path, output_path, num_samples=5, time_resolution=100, plot_show=True):
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
        if idx == 0:
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

# Execute Script
if __name__ == "__main__":

    test_name = "test02"
    phase_trans = "G2M_to_G1"
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    model_path = f"../../../models/interpolant/{test_name}_interp.pt"
    output_path = f"../../../results/interpolant/{test_name}/"

    run_diagnostics(model_path, data_path, output_path)
