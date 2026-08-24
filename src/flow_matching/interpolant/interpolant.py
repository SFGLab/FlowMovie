import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """Embeds a continuous scalar t in [0, 1] into a d_model vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PhiTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        
        # Projection for spatial features
        self.x0_proj = nn.Linear(input_dim, d_model)
        self.x1_proj = nn.Linear(input_dim, d_model)
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final head to output shape (D)
        self.out_head = nn.Linear(d_model, input_dim)

    def forward(self, x0, x1, t):
        # x0, x1: (Batch, D)
        # t: (Batch, 1) or (Batch,)
        
        # Embed inputs to sequence tokens: Length = 3 (x0, x1, time_token)
        h0 = self.x0_proj(x0).unsqueeze(1)    # (Batch, 1, d_model)
        h1 = self.x1_proj(x1).unsqueeze(1)    # (Batch, 1, d_model)
        ht = self.time_emb(t).unsqueeze(1)    # (Batch, 1, d_model)
        
        # Concatenate tokens into sequence: (Batch, 3, d_model)
        tokens = torch.cat([h0, h1, ht], dim=1)
        
        # Process through Multi-Head Attention
        feat = self.transformer(tokens)
        
        # Aggregate features across token sequence (or take the time token output)
        pooled = feat.mean(dim=1)  # (Batch, d_model)
        
        # Output Phi correction term: (Batch, D)
        phi = self.out_head(pooled)
        return phi

    