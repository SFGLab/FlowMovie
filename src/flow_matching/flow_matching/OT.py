import torch
import torch.nn as nn


def get_cost_matrix(x0, x1):
    """
    Computes a cost matrix to be used in the OT plan.
    Input: x0 and x1 of shape (B, D).
    Returns: cost_matrix of shape (B, B).
    """
    cost_matrix = torch.cdist(x0, x1, p=2) ** 2
    return cost_matrix


class SinkhornDistance(nn.Module):
    def __init__(self, eps=1e-4, reg=0.1, max_iter=100):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reg = reg

    def forward(self, cost_matrix, source, target):
        # cost_matrix shape: (B, B)
        # source shape: (B,)
        # target shape: (B,)

        M = torch.exp(-cost_matrix / self.reg)
        u = torch.ones_like(source)
        v = torch.ones_like(target)

        # Broadcasting replaces torch.diag(u) @ M @ torch.diag(v)
        P_prev = u.unsqueeze(1) * M * v.unsqueeze(0) 

        for _ in range(self.max_iter):
            u = source / (M @ v)
            v = target / (M.T @ u)

            P = u.unsqueeze(1) * M * v.unsqueeze(0)

            # Check convergence
            err = torch.linalg.norm(P_prev - P, "fro")
            if err < self.eps:
                break
            P_prev = P
            
        min_cost = torch.sum(P * cost_matrix)
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
