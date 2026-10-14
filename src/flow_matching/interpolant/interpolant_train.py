import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse

from interpolant_DataLoader import TrajectoryDataset
from interpolant import PhiTransformer


def compute_loss(mu, x_target, t, x0, x1, phi, d_target=1, vel_penalty_type = "l1",
                 lambda_dist=0.2, lambda_2d=0.1, lambda_vel=0.1):
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

    # 2D Coordinate Distance Loss (Last 2 features)
    mu_2d = mu[:, -2:]          # Shape: (B, 2)
    target_2d = x_target[:, -2:]  # Shape: (B, 2)
    
    # Euclidean distance (L2 norm) between predicted and target 2D points per sample
    dist_2d = torch.linalg.vector_norm(mu_2d - target_2d, dim=-1)  # Shape: (B,)
    loss_2d = torch.mean(dist_2d)

    # Velocity Loss
    velocity = (x1 - x0) + (1 - 2 * t) * phi
    if vel_penalty_type == "l2":
        velocity_loss = torch.mean(velocity ** 2)
    elif vel_penalty_type == "l1":
        velocity_loss = torch.mean(torch.abs(velocity))
    else:
        raise ValueError(f"Unknown penalty_type: {vel_penalty_type}")

    # Final combined loss
    # loss = mse_loss + (lambda_dist * distance_loss) + (lambda_2d * loss_2d)

    return mse_loss, lambda_dist * distance_loss, lambda_2d * loss_2d, lambda_vel * velocity_loss


def compute_random_target_loss(mu, x_target, d_target=1.0, lambda_dist=0.2, lambda_2d=0.1):
    """
    Computes loss against randomly shuffled targets to establish a baseline.
    """
    # Randomly permute the batch indices
    perm = torch.randperm(x_target.size(0), device=x_target.device)
    shuffled_target = x_target[perm]
    
    # Calculate loss against permuted targets
    return compute_loss(mu, shuffled_target, d_target, lambda_dist, lambda_2d)


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


def train_interpolant(pt_file_path, test_name="test01", num_epochs=20, delta_t = 0.01, lr=1e-5):
    if torch.cuda.is_available():
        print("Running on CUDA...")
        device = torch.device("cuda")
    else:
        print("Running on CPU (CUDA not available)...")
        device = torch.device("cpu")
    dataset = TrajectoryDataset(pt_file_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    input_dim = dataset.metadata["num_values"]  # D
    num_frames = dataset.metadata["num_frames"] # N
    
    model = PhiTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    avg_losses = []
    model.train()
    for epoch in range(num_epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss4 = 0.0
                
        # single_batch = next(iter(dataloader))
        # single_batch = single_batch.to(device)
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
            loss1, loss2, loss3, loss4 = compute_loss(mu, x_target, t, x0, x1, phi, 
                                               d_target=1, lambda_dist=0.2, lambda_2d=1.7, lambda_vel=0.9)
            # loss1, loss2, loss3 = compute_random_target_loss(mu, x_target, d_target=1, lambda_dist=0.1, lambda_2d=0.2)
                        
            loss = loss1 + loss2 + loss3 + loss4
            # metrics = compute_metrics(mu, v_est, x_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss4 += loss4.item()         
            
        avg_loss1 = total_loss1 / len(dataloader)
        avg_loss2 = total_loss2 / len(dataloader)
        avg_loss3 = total_loss3 / len(dataloader)
        avg_loss4 = total_loss4 / len(dataloader)     
                
        avg_losses.append(avg_loss1 + avg_loss2 + avg_loss3)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss1:.2f} + {avg_loss2:.2f} + {avg_loss3:.2f}  + {avg_loss4:.2f}")

    # Save losses:
    losses_path = f"../../../results/interpolant/{test_name}/losses"
    if not os.path.exists(losses_path):
        os.makedirs(losses_path)
    np.savetxt(os.path.join(losses_path, f"{test_name}_loss_train.txt"), avg_losses)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train an interpolant model and save the checkpoint."
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
        "--num-epochs",
        "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../../data/md",
        help="Base directory for processed data (default: '../../../data/md')",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="../../../models/interpolant",
        help="Base directory for saving models (default: '../../../models/interpolant')",
    )

    args = parser.parse_args()

    data_path = f"{args.data_dir}/{args.test_name}/processed/{args.phase_trans}_data.pt"
    model_path = f"{args.model_dir}/{args.test_name}_interp_model.pt"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Execute Training
    trained_model = train_interpolant(data_path, test_name=args.test_name, num_epochs=args.num_epochs)

    # Save trained model
    checkpoint = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": trained_model.input_dim,  # e.g., D = K*3+2
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    main()

