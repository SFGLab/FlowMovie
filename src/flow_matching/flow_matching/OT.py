import torch
import torch.nn as nn

# from src.flow_matching.flow_matching.FM_train import compute_flow_matching_target


def get_cost_matrix(x0, x1):
    """
    Computes a cost matrix to be used in the OT plan.
    Input: x0 and x1 of shape (B, D).
    Returns: cost_matrix of shape (B, B).
    """
    cost_matrix = torch.cdist(x0[:,:-2], x1[:,:-2], p=2) ** 2
    return cost_matrix

def get_cost_matrix_msd(x0, x1):
    """
    Computes a cost matrix to be used in the OT plan as a MSD.
    Input: x0 and x1 of shape (B, D).
    Returns: cost_matrix of shape (B, B).
    """
    coords0 = x0[:, :-2]
    coords1 = x1[:, :-2]
    K = coords0.shape[1] // 3
    
    # Squared cdist divided by K
    msd_matrix = (torch.cdist(coords0, coords1, p=2) ** 2) / K
    
    return msd_matrix

# def get_trajectory_cost_matrix_xt(phi_model, x0, x1, num_steps=10):
#     B = x0.shape[0]
#     device = x0.device
#     K = (x0.shape[1] - 2) // 3  # Number of 3D points
    
#     # 1. Expand to evaluate all B x B combinations
#     # x0_rep: [x0_0, x0_0, ..., x0_1, x0_1, ...]
#     x0_rep = x0.repeat_interleave(B, dim=0) 
#     # x1_rep: [x1_0, x1_1, ..., x1_0, x1_1, ...]
#     x1_rep = x1.repeat(B, 1)

#     # 2. Define time steps
#     t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    
#     total_cost = torch.zeros(B * B, device=device)
    
#     # Get the starting point at t=0
#     # (Assuming your function accepts a batched tensor for t)
#     t_tensor = torch.full((B * B,), t_vals[0].item(), device=device)
#     xt_prev, _ = compute_flow_matching_target(phi_model, x0_rep, x1_rep, t_tensor)
#     coords_prev = xt_prev[:, :-2]
    
#     # 3. Step through time and accumulate MSD
#     for i in range(1, len(t_vals)):
#         t_tensor = torch.full((B * B,), t_vals[i].item(), device=device)
#         xt_curr, _ = compute_flow_matching_target(phi_model, x0_rep, x1_rep, t_tensor)
#         coords_curr = xt_curr[:, :-2]
        
#         # Calculate MSD for this time step
#         step_msd = torch.sum((coords_curr - coords_prev) ** 2, dim=1) / K
#         total_cost += step_msd
        
#         coords_prev = coords_curr
        
#     # 4. Reshape back to a (B, B) cost matrix
#     cost_matrix = total_cost.view(B, B)
#     return cost_matrix


# def get_trajectory_cost_matrix_ut(phi_model, x0, x1, num_steps=10):
#     B = x0.shape[0]
#     device = x0.device
#     K = (x0.shape[1] - 2) // 3
    
#     x0_rep = x0.repeat_interleave(B, dim=0) 
#     x1_rep = x1.repeat(B, 1)
    
#     # Time steps
#     t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)
#     dt = 1.0 / num_steps
    
#     total_cost = torch.zeros(B * B, device=device)
    
#     for t in t_vals:
#         t_tensor = torch.full((B * B,), t.item(), device=device)
        
#         # We only need ut here
#         _, ut = compute_flow_matching_target(phi_model, x0_rep, x1_rep, t_tensor)
#         u_coords = ut[:, :-2]
        
#         # Velocity squared (Kinetic Energy / Action) normalized by K
#         # MSD of the velocity field at time t
#         u_msd = torch.sum(u_coords ** 2, dim=1) / K
        
#         # Integrate over time: cost += energy * dt
#         total_cost += u_msd * dt
        
#     return total_cost.view(B, B)


class SinkhornDistance(nn.Module):
    def __init__(self, eps=1e-4, reg=0.1, max_iter=100):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reg = reg

    def forward(self, cost_matrix, source, target):
        cost_matrix_unscaled = cost_matrix

        # Normalize the cost matrix to prevent underflow
        max_val = torch.max(cost_matrix)
        if max_val > 0:
            cost_matrix = cost_matrix / max_val

        M = torch.exp(-cost_matrix / self.reg)
        u = torch.ones_like(source)
        v = torch.ones_like(target)

        P_prev = u.unsqueeze(1) * M * v.unsqueeze(0) 

        for _ in range(self.max_iter):
            # Added small epsilon (1e-8) to denominators to prevent division by zero
            u = source / (M @ v + 1e-8)
            v = target / (M.T @ u + 1e-8)

            P = u.unsqueeze(1) * M * v.unsqueeze(0)

            # Check convergence
            err = torch.linalg.norm(P_prev - P, "fro")
            if err < self.eps:
                break
            P_prev = P 

        # Compute min_cost using the original unscaled cost matrix
        min_cost = torch.sum(P * cost_matrix_unscaled)
        return P, min_cost


def get_OT_x1(x0, x1, eps=1e-4, reg=0.1, max_iter=100):
    """
    Calculates the cost matrix between observations from x0 and x1
    and uses it to calculate the OT plan. 
    Returns the modified x1 for which each row aligns with the optimal row from x0.
    Rows in new x1 may repeat.
    """

    # Compute the Cost Matrix (Squared Euclidean Distance)
    # shape: (B, B)
    cost_matrix = get_cost_matrix(x0, x1)
    # Define uniform marginal distributions
    B = x0.shape[0]
    a = torch.ones(B) / B  # source weights
    b = torch.ones(B) / B  # target weights

    # Run Sinkhorn
    sinkhorn = SinkhornDistance(eps=eps, reg=reg, max_iter=max_iter)
    P, min_cost = sinkhorn(cost_matrix, a, b)

    # SOFT SHUFFLE (Barycentric Projection)
    # Scales P by B so weights sum to 1, then multiplies by x1
    # x1_soft shape: (B, D)
    x1_soft = (P * B) @ x1
    return x1_soft
