import tqdm
import math
import torch
import numpy as np
from torch import nn


# class Block(nn.Module):
#     def __init__(self, channels=512):
#         super().__init__()
#         self.ff = nn.Linear(channels, channels)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         return self.act(self.ff(x))

# class MLP(nn.Module):
#     def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
#         super().__init__()
#         self.channels_data = channels_data
#         self.layers = layers
#         self.channels = channels
#         self.channels_t = channels_t
#         self.in_projection = nn.Linear(channels_data, channels)
#         self.t_projection = nn.Linear(channels_t, channels)
#         self.blocks = nn.Sequential(*[
#             Block(channels) for _ in range(layers)
#         ])
#         self.out_projection = nn.Linear(channels, channels_data)

#     def gen_t_embedding(self, t, max_positions=10000):
#         # Converting [B, 1] -> [B] if necessary
#         if t.ndim == 2:
#             t = t.squeeze(1) 
            
#         t = t * max_positions
#         half_dim = self.channels_t // 2
#         emb = math.log(max_positions) / (half_dim - 1)
#         emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([emb.sin(), emb.cos()], dim=1) # [B, channels_t]
        
#         if self.channels_t % 2 == 1:
#             emb = nn.functional.pad(emb, (0, 1), mode='constant')
#         return emb

#     def forward(self, x, t):
#         x = self.in_projection(x)
#         t = self.gen_t_embedding(t)
#         t = self.t_projection(t)
#         x = x + t
#         x = self.blocks(x)
#         x = self.out_projection(x)
#         return x


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.dim = dim
        # Fixed random Gaussian frequencies
        self.W = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # Scale t to [0, 2*pi] frequency projections
        freqs = t * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels)
        self.t_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        res = x
        x = self.norm(x)
        # Inject time conditioning into hidden features
        h = self.act(self.in_proj(x) + self.t_proj(t_emb))
        h = self.act(self.out_proj(h))
        return res + h


class ContinuousFlowMLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_data = channels_data
        self.layers = layers
        self.channels = channels
        self.channels_t = channels_t
        self.t_embedder = FourierTimeEmbedding(channels_t, scale=16.0)
        self.t_mlp = nn.Sequential(
            nn.Linear(channels_t, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        self.in_projection = nn.Linear(channels_data, channels)
        
        self.blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(layers)
        ])
        
        self.out_projection = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels_data)
        )

    def forward(self, x, t):
        # 1. Smooth time embedding
        t_emb = self.t_embedder(t)
        t_emb = self.t_mlp(t_emb)
        
        # 2. Input projection
        x = self.in_projection(x)
        
        # 3. Pass through residual blocks with per-layer time conditioning
        for block in self.blocks:
            x = block(x, t_emb)
            
        # 4. Output projection
        return self.out_projection(x)
    

def load_FM_model(model_path, device):
    """
    Function for loading of the trained FM model.
    """
    checkpoint = torch.load(model_path, map_location=device)
    FM_model = ContinuousFlowMLP(
                    channels_data=checkpoint["channels_data"], 
                    layers=checkpoint["layers"], 
                    channels=checkpoint["channels"],
                    channels_t=checkpoint["channels_t"]).to(device)
    FM_model.load_state_dict(checkpoint["model_state_dict"])
    return FM_model

