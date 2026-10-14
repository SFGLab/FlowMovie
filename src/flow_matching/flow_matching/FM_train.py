import os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

from src.flow_matching.interpolant.interpolant import load_phi_model
from src.flow_matching.interpolant.interpolant_DataLoader import TrajectoryDataset
from src.flow_matching.flow_matching.OT import get_OT_x1
from src.flow_matching.flow_matching.FM_model import ContinuousFlowMLP


def compute_flow_matching_xt(phi_model, x0, x1, t):
    """
    Computes and returns mu(t).
    """
    t = t.clone().detach().requires_grad_(True)
    phi = phi_model(x0, x1, t.squeeze(1))
    return (1 - t) * x0 + t * x1 + t * (1 - t) * phi


def compute_flow_matching_target(phi_model, x0, x1, t):
    """
    Computes xt and the velocity target ut for Flow Matching training.
    
    x0: (Batch, K*3)
    x1: (Batch, K*3)
    t:  (Batch, 1) in [0, 1]
    """
    # Enable gradient tracking for continuous scalar time t
    t = t.clone().detach().requires_grad_(True) # (Batch, 1)
    
    # Predict Phi(x0, x1, t)
    phi = phi_model(x0, x1, t.squeeze(1)) # (Batch, K*3)
    
    # Compute continuous interpolated path xt
    xt = (1 - t) * x0 + t * x1 + t * (1 - t) * phi # (Batch, K*3)
    
    # Compute partial derivative d(Phi) / dt using autograd
    # Summing over output components lets autograd compute batch-wise vector-Jacobian products
    dphi_dt = torch.zeros_like(phi)
    for k in range(phi.shape[1]):
        grad_k = torch.autograd.grad(
            outputs=phi[:, k].sum(),
            inputs=t,
            create_graph=True, # Set True if velocity model v_theta backprops into Phi
            retain_graph=True
        )[0] # Shape: (Batch, 1)
        dphi_dt[:, k] = grad_k.squeeze(1)

    # Compute true velocity target ut = d(xt) / dt
    target_ut = (x1 - x0) + (1 - 2 * t) * phi + t * (1 - t) * dphi_dt
    
    return xt.detach(), target_ut.detach()


def compute_flow_matching_target_fast(phi_model, x0, x1, t):
    """
    Vectorized computation of d(Phi)/dt for high-dimensional feature spaces.

    x0: (Batch, D)
    x1: (Batch, D)
    t:  (Batch, 1) in [0, 1]
    """
    t = t.clone().detach().requires_grad_(True) # (Batch, 1)
    
    # Forward pass
    phi = phi_model(x0, x1, t.squeeze(1)) # (Batch, D)
    
    # Compute d(Phi)/dt using autograd.grad with a dummy vector product
    # Since Phi only depends on t scalar per batch item:
    grad_outputs = torch.ones_like(phi)
    dphi_dt_sum = torch.autograd.grad(
        outputs=phi,
        inputs=t,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False
    )[0] # (Batch, 1)
    
    # Expand dphi_dt across feature dimensions
    # Note: If Phi_i depends only on t_i for the same batch index:
    # d(Phi)/dt_i = autograd gradient scaled across features
    
    # Compute Path xt
    xt = (1 - t) * x0 + t * x1 + t * (1 - t) * phi
    
    # Compute exact analytic target ut
    # (Assuming Phi output features share time derivative d(Phi)/dt)
    target_ut = (x1 - x0) + (1 - 2 * t) * phi + t * (1 - t) * dphi_dt_sum
    
    return xt.detach(), target_ut.detach()


def get_trajectory(phi_model, x0, x1, num_steps=10):
    """
    Computes num_steps structures based on phi_model between x0 and x1.
    Uses evenly distributed times between 0 and 1.
    Returns a list of structures of length num_steps from x0 to x1.
    """
    time_steps = torch.linspace(0, 1, steps=num_steps, device=x0.device)
    steps = []
    for t in time_steps:
        time_step = torch.full((x0.shape[0],), t, device=x0.device)[:, None]
        xt = compute_flow_matching_xt(phi_model, x0, x1, time_step)
        steps.append(xt)
    return steps


def visualize_batch_OT_plan(x0, x1, x1_OT, batch_id, output_path, phi_model, num_steps=10):
    """
    Creates a plot of the pre-OT and post-OT interpolant paths based on phi_model.
    """
    steps_preOT = get_trajectory(phi_model, x0, x1, num_steps=num_steps)
    steps_postOT = get_trajectory(phi_model, x0, x1_OT, num_steps=num_steps)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(x0.shape[0]):
        traj_x = [step[i,-2].cpu().detach().numpy() for step in steps_preOT]
        traj_y = [step[i,-1].cpu().detach().numpy() for step in steps_preOT]
        ax[0].plot(traj_x, traj_y, color="red", alpha=0.2, lw=5)
        traj_x = [step[i,-2].cpu().detach().numpy() for step in steps_postOT]
        traj_y = [step[i,-1].cpu().detach().numpy() for step in steps_postOT]
        ax[1].plot(traj_x, traj_y, color="red", alpha=0.2, lw=5)
        
    ax[0].set_title(
        f"Trajectories pre OT",
        fontsize=14,
        fontweight="bold",
    )
    ax[1].set_title(
        f"Trajectories post OT",
        fontsize=14,
        fontweight="bold",
    )
    for ax_id in [0,1]:
        ax[ax_id].set_xlabel("X")
        ax[ax_id].set_ylabel("Y")
        ax[ax_id].grid(True, linestyle="--", alpha=0.5)

    filename = os.path.join(output_path, f"OT_trajectories_batch{str(batch_id).zfill(3)}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def visualize_velocity_correlation(
    tensor: torch.Tensor,
    t: torch.Tensor,
    output_path: str,
    xlabel: str = "X velocity",
    ylabel: str = "Y velocity",
    clabel: str = "Time (t)",
    cmap: str = "viridis",
    title: str = "Velocity relationship between RNA-seq coordinates.",
):
    """Plots the last column against the second-to-last column of a (B, D) PyTorch tensor.

    Args:
        tensor (torch.Tensor): 2D PyTorch tensor of shape (B, D) where D >= 2.
        output_path: Path where the plot should be saved.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        clabel (str): Label for the colorbar.
        cmap (str): Matplotlib colormap name.
        title (str): Title for the plot.
    """
    if tensor.ndim != 2 or tensor.shape[1] < 2:
        raise ValueError(
            "Expected a 2D tensor with shape (B, D) where D >= 2."
        )

    x = tensor[:, -2].detach().cpu().view(-1).numpy()
    y = tensor[:, -1].detach().cpu().view(-1).numpy()
    c = t.detach().cpu().view(-1).numpy()

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(x, y, c=c, cmap=cmap, alpha=0.8, edgecolors="none")

    cbar = plt.colorbar(scatter)
    cbar.set_label(clabel)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    filename = os.path.join(output_path)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def calculate_FM_loss(prediction, target_ut, reg_lambda=0.01, penalty_type="l2"):
    """
    Calculates Flow Matching loss with velocity magnitude regularization.

    Args:
        prediction (torch.Tensor): Model velocity prediction, shape (B, D)
        target_ut (torch.Tensor): Target velocity vector, shape (B, D)
        reg_lambda (float): Regularization strength weight (e.g., 1e-3 to 1e-1)
        penalty_type (str): 'l2' for Kinetic Energy, 'l1' for peak dampening

    Returns:
        torch.Tensor: Scalar loss tensor
    """
    # Standard Flow Matching MSE loss
    fm_loss = ((prediction - target_ut) ** 2).mean()

    # Regularization penalty on velocity magnitudes
    if penalty_type == "l2":
        # Kinetic Energy Penalty: sum over dimensions D, average over batch B
        reg_loss = (prediction**2).sum(dim=-1).mean()
    elif penalty_type == "l1":
        # L1 Norm Penalty: dampens extreme velocity spikes more aggressively
        reg_loss = torch.abs(prediction).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    total_loss = fm_loss + reg_lambda * reg_loss
    return total_loss


def train_FM(FM_model, phi_model_path, optim, pt_file_path, test_name, mode="phi", training_steps=500, batch_size=10, 
             reg_lambda=0.01, penalty_type="l2", OT_reg=0.1, viz_OT=False):
    """
    Main function for the execution of the training of the FM model.
    Returns losses for the training and test set.
    """
    if torch.cuda.is_available():
        print("Running on CUDA...")
        device = torch.device("cuda")
    else:
        print("Running on CPU (CUDA not available)...")
        device = torch.device("cpu")

    dataset = TrajectoryDataset(pt_file_path)
    full_batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=full_batch_size, shuffle=True)
    batch_trajectories = next(iter(dataloader)).to(device)
    # batch_trajectories shape: (Total_Dataset_Size, N, D)
    B, N, D = batch_trajectories.shape
    data_x0 = batch_trajectories[:, 0, :]   # First frame (B, D)
    data_x1 = batch_trajectories[:, -1, :]  # Last frame (B, D)
    
    # Split indices (80/20)
    indices = torch.randperm(B)
    train_size = int(0.8 * B)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    # Slice tensors
    x0_train, x1_train = data_x0[train_idx], data_x1[train_idx]
    x0_test, x1_test = data_x0[test_idx], data_x1[test_idx]

    # prepare interpolant Phi
    phi_model = load_phi_model(phi_model_path, device)
    phi_model.eval()

    # main training loop
    pbar = tqdm(range(training_steps))
    losses_train = []
    losses_test = []
    for i in pbar:
        # ======== calculate train loss ======== 
        ids = torch.randint(x0_train.size(0), (batch_size,))
        x0 = x0_train[ids]
        x1 = x1_train[ids]
        t = torch.rand((x1.size(0), 1), device=device)

        # Optimal Transport plan (modifies x1 so as to create OT):
        x1_OT = get_OT_x1(x0, x1, reg=OT_reg)
        
        # target velocity and xt with or without Phi interpolant correction:
        if mode == "linear":
            target_ut = x1_OT - x0
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1_OT
        elif mode == "phi":
            xt, target_ut = compute_flow_matching_target_fast(phi_model, x0, x1_OT, t)

        pred = FM_model(xt, t)
        loss_train = calculate_FM_loss(pred, target_ut, reg_lambda=reg_lambda, penalty_type=penalty_type)

        # Visualization of the OT and velocities
        if viz_OT and i == 0:
            print("\nVisualizing the OT plan for the first iteration...")
            viz_path = f"../../../results/FM/{test_name}/viz"
            if not os.path.exists(viz_path): os.makedirs(viz_path)
            visualize_batch_OT_plan(x0, x1, x1_OT, i, \
                                    viz_path, phi_model, num_steps=N)
            
            visualize_velocity_correlation(target_ut, t, 
                                           os.path.join(viz_path, "FM_velocities_mu.png"))
            visualize_velocity_correlation(pred, t, 
                                           os.path.join(viz_path, "FM_velocities_fm.png"))
        
        # ======== calculate test loss ======== 
        # Optimal Transport plan
        x1_test_OT = get_OT_x1(x0_test, x1_test, reg=OT_reg)

        t = torch.rand((x1_test.size(0), 1), device=device)

        if mode == "linear":
            target_ut = x1_test - x1_test_OT
            xt = (1 - t[:, None]) * x1_test_OT + t[:, None] * x1_test
        elif mode == "phi":
            xt, target_ut = compute_flow_matching_target_fast(phi_model, x1_test_OT, x1_test, t)
        
        pred = FM_model(xt, t)
        loss_test = calculate_FM_loss(pred, target_ut, reg_lambda=reg_lambda, penalty_type=penalty_type)
    
        # ======== train ======== 
        loss_train.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss_train.item())
        losses_train.append(loss_train.item())
        losses_test.append(loss_test.item())

    return losses_train, losses_test


def main(test_name, phase_trans, mode="phi", training_steps=100, batch_size=10, 
         layers=5, channels_h=512, channels_t=512, 
         reg_lambda=0.01, penalty_type="l2", OT_reg=0.1, lr=1e-3, viz_OT=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi_model_path = f"../../../models/interpolant/{test_name}_interp_model.pt"
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    D = TrajectoryDataset(data_path).get_D() # data dimension e.g. D = 3*K + 2

    FM_model = ContinuousFlowMLP(channels_data=D, layers=layers, channels=channels_h, channels_t=channels_t).to(device)
    optim = torch.optim.AdamW(FM_model.parameters(), lr=lr)

    # Execute training
    losses_train, losses_test = train_FM(FM_model, phi_model_path, optim, data_path, test_name=test_name,
                                mode=mode, training_steps=training_steps, 
                                batch_size=batch_size, reg_lambda=reg_lambda, penalty_type=penalty_type,
                                OT_reg=OT_reg, viz_OT=viz_OT)

    # Save losses:
    losses_path = f"../../../results/FM/{test_name}/losses"
    if not os.path.exists(losses_path):
        os.makedirs(losses_path)
    np.savetxt(os.path.join(losses_path, f"{test_name}_loss_train.txt"), losses_train)
    np.savetxt(os.path.join(losses_path, f"{test_name}_loss_test.txt"), losses_test)

    # Save trained model:
    FM_model_dir = f"../../../models/FM"
    FM_model_path = os.path.join(FM_model_dir, f"{test_name}_FM_model.pt")
    if not os.path.exists(FM_model_dir):
        os.makedirs(FM_model_dir)
    checkpoint = {
        "model_state_dict": FM_model.state_dict(),
        "channels_data": FM_model.channels_data,
        "layers": FM_model.layers,
        "channels": FM_model.channels,
        "channels_t": FM_model.channels_t
    }
    torch.save(checkpoint, FM_model_path)
    print(f"Model saved successfully to {FM_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run training pipeline with optimal transport (OT) regularization."
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
        "--mode",
        "-m",
        type=str,
        default="phi",
        help="Interpolation mode (default: 'phi')",
    )
    parser.add_argument(
        "--training-steps",
        "-s",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Batch size (default: 50)",
    )
    parser.add_argument(
        "--layers",
        "-l",
        type=int,
        default=5,
        help="Number of NN layers (default: 5)",
    )
    parser.add_argument(
        "--channels-h",
        "-ch",
        type=int,
        default=512,
        help="Number of channels in hidden NN layers (default: 512)",
    )
    parser.add_argument(
        "--channels-t",
        "-ct",
        type=int,
        default=512,
        help="Number of channels in time embedding NN layers (default: 512)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the NN optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--fm-reg",
        "-fr",
        type=float,
        default=0.01,
        help="FM velocity regularization parameter (default: 0.01)",
    )
    parser.add_argument(
        "--fm-reg-mode",
        "-fm",
        type=str,
        default="l2",
        help="FM velocity regularization mode (default: 'l2')",
    )
    parser.add_argument(
        "--ot-reg",
        "-r",
        type=float,
        default=0.01,
        help="OT regularization parameter (default: 0.01)",
    )
    parser.add_argument(
        "--viz-ot",
        action="store_true",
        help="Enable OT visualization (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        args.test_name,
        args.phase_trans,
        mode=args.mode,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        OT_reg=args.ot_reg,
        layers=args.layers, 
        channels_h=args.channels_h, 
        channels_t=args.channels_t, 
        lr=args.learning_rate,
        reg_lambda=args.fm_reg,
        penalty_type=args.fm_reg_mode,
        viz_OT=args.viz_ot,
    )

    