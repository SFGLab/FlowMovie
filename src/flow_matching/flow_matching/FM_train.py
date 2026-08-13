import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.flow_matching.interpolant.interpolant import PhiTransformer
from src.flow_matching.interpolant.interpolant_diagnostics import predict_mu
from src.flow_matching.flow_matching.OT import SinkhornDistance
from src.flow_matching.flow_matching.FM_model import MLP
from src.flow_matching.interpolant.interpolant_DataLoader import TrajectoryDataset


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
    """
    t = t.clone().detach().requires_grad_(True) # (Batch, 1)
    
    # Forward pass
    phi = phi_model(x0, x1, t.squeeze(1)) # (Batch, K*3)
    
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


def train_FM(model, phi_model_path, optim, pt_file_path, mode="phi", training_steps=500, batch_size=10):
    if torch.cuda.is_available():
        print("Running on CUDA...")
        device = torch.device("cuda")
    else:
        print("Running on CPU...")
        device = torch.device("cpu")
    dataset = TrajectoryDataset(pt_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_trajectories in dataloader:
        # batch_trajectories shape: (Batch_Size, N, D)
        batch_trajectories = batch_trajectories.to(device)
        B, N, D = batch_trajectories.shape
        x0 = batch_trajectories[:, 0, :]   # First frame (B, D)
        x1 = batch_trajectories[:, -1, :]  # Last frame (B, D)

    data_x0 = x0
    data_x1 = x1
    
    indices = torch.randperm(batch_size)
    
    # Split indices (80/20)
    train_size = int(0.8 * batch_size)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    # Slice tensors
    x0_train, x1_train = data_x0[train_idx], data_x1[train_idx]
    x0_test, x1_test = data_x0[test_idx], data_x1[test_idx]

    # prepare interpolant Phi
    checkpoint = torch.load(phi_model_path, map_location=device)
    phi_model = PhiTransformer(input_dim=checkpoint["input_dim"]).to(device)
    phi_model.load_state_dict(checkpoint["model_state_dict"])
    phi_model.eval()

    # main training loop
    pbar = tqdm.tqdm(range(training_steps))
    losses_train = []
    losses_test = []
    for i in pbar:
        # ======== calculate train loss ======== 
        ids = torch.randint(x0_train.size(0), (batch_size,))
        x0 = x0_train[ids]
        x1 = x1_train[ids]
        t = torch.rand((x1.size(0), 1), device=device)

        # OT here (modifies x0 and x1 so as to create OT):
        # TODO 
 
        # target velocity and xt with or without Phi interpolant correction:
        if mode == "linear":
            target_ut = x1 - x0
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        elif mode == "phi":
            xt, target_ut = compute_flow_matching_target(phi_model, x0, x1, t)

        pred = model(xt, t)
        loss_train = ((target_ut - pred)**2).mean()
    
        # ======== calculate test loss ======== 
        # OT TODO

        t = torch.rand((x1_test.size(0), 1), device=device)

        if mode == "linear":
            target_ut = x1_test - x0_test
            xt = (1 - t[:, None]) * x0_test + t[:, None] * x1_test
        elif mode == "phi":
            xt, target_ut = compute_flow_matching_target(phi_model, x0_test, x1_test, t)
        
        pred = model(xt, t)
        loss_test = ((target_ut - pred)**2).mean()
    
        # ======== train ======== 
        loss_train.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss_train.item())
        losses_train.append(loss_train.item())
        losses_test.append(loss_test.item())

    return losses_train, losses_test


if __name__ == "__main__":
    # Execute training

    model = MLP(layers=5, channels_data=7976, channels=512)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    test_name = "test02"
    phase_trans = "G2M_to_G1"
    phi_model_path = f"../../../models/interpolant/{test_name}_interp.pt"
    data_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    
    losses_train, losses_test = train_FM(model, phi_model_path, optim, data_path,
                                mode="phi", training_steps=4, batch_size=5)

    # Save trained model:
    model_path = f"../../../models/FM/{test_name}_FM_model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "channels_data": model.channels_data,
        "layers": model.layers,
        "channels": model.channels,
        "channels_t": model.channels_t
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved successfully to {model_path}")

    