import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from interpolant_DataLoader import TrajectoryDataset
from interpolant import PhiTransformer


def compute_loss(mu, x_target, d_target=1, lambda_dist=0.2):
    """
    Calculates the loss function for the interpolant.
    mu is expected to be: mu = (1 - t) * x0 + t * x1 + t * (1 - t) * phi
    x_target is the ChromMovie structure with which the MSE is calculated.
    """

    # MSE 
    criterion = nn.MSELoss()
    mse_loss = criterion(mu, x_target)

    # Consecutive bead distance loss
    B = mu.size(0)
    K = (mu.size(1)-2) // 3
    mu_3d = mu[:,:-2].view(B, K, 3)
    diffs = mu_3d[:, 1:, :] - mu_3d[:, :-1, :] # Shape: (B, K-1, 3)
    distances = torch.linalg.vector_norm(diffs, dim=-1) # Shape: (B, K-1)
    distance_loss = torch.mean((distances - d_target) ** 2)

    # Final loss
    loss = mse_loss + lambda_dist * distance_loss

    return loss


def compute_metrics(mu, mu_prev, x_target):
    """
    Calculates a number of metrics for the current state of the interpolant.
    """
    B = mu.size(0)
    K = (mu.size(1)-2) // 3

    mu_3d = mu[:,:-2].view(B, K, 3)
    print(mu.shape, mu_3d.shape)
    print(mu[0, :6])
    print(mu_3d[:2, :])

    return


def train_interpolant(pt_file_path, num_epochs=20, delta_t = 0.01, lr=1e-3):
    if torch.cuda.is_available():
        print("Running on CUDA...")
        device = torch.device("cuda")
    else:
        print("Running on CPU (CUDA not available)...")
        device = torch.device("cpu")
    dataset = TrajectoryDataset(pt_file_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    input_dim = dataset.metadata["num_values"]  # D
    num_frames = dataset.metadata["num_frames"] # N
    
    model = PhiTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    avg_losses = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_trajectories in dataloader:
            # batch_trajectories shape: (Batch_Size, N, D)
            batch_trajectories = batch_trajectories.to(device)
            B, N, D = batch_trajectories.shape
            
            x0 = batch_trajectories[:, 0, :]   # First frame (B, D)
            x1 = batch_trajectories[:, -1, :]  # Last frame (B, D)
            
            # Draw continuous float times t ~ Uniform(0, 1) for each trajectory
            t = torch.rand((B, 1), device=device) # Shape: (B, 1), values in [0.0, 1.0)
            # t1 = torch.rand((B, 1), device=device) * (1.0 - delta_t)
            # t2 = t1 + delta_t
            # x0_cat = torch.cat([x0, x0], dim=0)
            # x1_cat = torch.cat([x1, x1], dim=0)
            # t_cat = torch.cat([t1, t2], dim=0).squeeze(1)

            # Map continuous time t to closest integer frame indices in range [0, N - 1]
            frame_indices = torch.round(t * (N - 1)).long().squeeze(1) # Shape: (B,)
            
            # Extract ground truth target frames x_target
            x_target = batch_trajectories[torch.arange(B), frame_indices, :] # (B, D)

            # Predict Phi and mu
            phi = model(x0, x1, t.squeeze(1)) # (B, D)
            # phi_cat = model(x0_cat, x1_cat, t_cat)
            # phi1, phi2 = torch.chunk(phi_cat, 2, dim=0)

            mu = (1 - t) * x0 + t * x1 + t * (1 - t) * phi # (B, D)
            # mu1 = (1 - t1) * x0 + t1 * x1 + t1 * (1 - t1) * phi1
            # mu2 = (1 - t2) * x0 + t2 * x1 + t2 * (1 - t2) * phi2
            # v_est = (mu2 - mu1) / delta_t

            # Compute loss and other metrics
            loss = compute_loss(mu, x_target)
            # metrics = compute_metrics(mu, v_est, x_target)
            # return
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Save losses:
    losses_path = f"../../../results/interpolant/{test_name}/losses"
    if not os.path.exists(losses_path):
        os.makedirs(losses_path)
    np.savetxt(os.path.join(losses_path, f"{test_name}_loss_train.txt"), avg_losses)

    return model


if __name__ == "__main__":
    # Execute Training
    test_name = "test02"
    phase_trans = "G2M_to_G1"
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    trained_model = train_interpolant(data_path, num_epochs=150)

    # Save trained model:
    model_path = f"../../../models/interpolant/{test_name}_interp_model.pt"
    checkpoint = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": trained_model.input_dim,  # e.g., D = K*3+2
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved successfully to {model_path}")
    