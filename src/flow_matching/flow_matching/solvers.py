import tqdm
import math
import torch
import numpy as np
from torch import nn


@torch.no_grad()
def iterate_trajectory_pytorch(model, x0, num_steps=100, save_every=10, method="rk4"):
    """
    x0: torch.Tensor of shape (3m,) or (m, 3)
    """
    device = x0.device if isinstance(x0, torch.Tensor) else torch.device("cpu")
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, dtype=torch.float32, device=device)
        
    t_span = torch.linspace(0.0, 1.0, num_steps, device=device)
    dt = t_span[1] - t_span[0]
    
    x_curr = x0.clone()
    structures = [x_curr.cpu().numpy()]
    
    model.eval()
    for i in range(len(t_span) - 1):
        t = t_span[i]
        
        if method == "euler":
            # Simple Forward Euler: x_next = x + v * dt
            v = model(x_curr, t.unsqueeze(0))
            x_curr = x_curr + v * dt
            
        elif method == "rk4":
            # 4th Order Runge-Kutta for higher precision
            k1 = model(x_curr, t.unsqueeze(0))
            k2 = model(x_curr + 0.5 * dt * k1, (t + 0.5 * dt).unsqueeze(0))
            k3 = model(x_curr + 0.5 * dt * k2, (t + 0.5 * dt).unsqueeze(0))
            k4 = model(x_curr + dt * k3, (t + dt).unsqueeze(0))
            
            x_curr = x_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if (i + 1) % save_every == 0:
            structures.append(x_curr.cpu().numpy())
            
    return structures