import tqdm
import math
import torch
import numpy as np
from torch import nn


class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_data = channels_data
        self.layers = layers
        self.channels = channels
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[
            Block(channels) for _ in range(layers)
        ])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t, max_positions=10000):
        # Converting [B, 1] -> [B] if necessary
        if t.ndim == 2:
            t = t.squeeze(1) 
            
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1) # [B, channels_t]
        
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x


def load_FM_model(model_path, device):
    """
    Function for loading of the trained FM model.
    """
    checkpoint = torch.load(model_path, map_location=device)
    FM_model = MLP(layers=checkpoint["layers"], 
                    channels_data=checkpoint["channels_data"], 
                    channels=checkpoint["channels"]).to(device)
    FM_model.load_state_dict(checkpoint["model_state_dict"])
    return FM_model

